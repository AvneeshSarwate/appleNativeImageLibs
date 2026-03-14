import SwiftUI
import AVFoundation
import CoreML
import CoreVideo
import Shared

// ============================================================================
// MARK: - Configuration (swap these to test different setups)
// ============================================================================

struct PipelineConfig {
    let yoloPath: String
    let posePath: String
    let yoloUnits: MLComputeUnits
    let poseUnits: MLComputeUnits
    let poseH: Int
    let poseW: Int
    let parallel: Bool  // true = YOLO+Pose run concurrently with 1-frame bbox lag

    // --- Presets ---

    /// Best for TD coexistence: YOLO on ANE, DWPose-m on GPU, parallel
    static let parallelANEGPU = PipelineConfig(
        yoloPath: "yolo11s-seg.mlpackage", posePath: "dwpose_m.mlpackage",
        yoloUnits: .all, poseUnits: .cpuAndGPU,
        poseH: 256, poseW: 192, parallel: true)

    /// Zero GPU: YOLO on ANE, DWPose-m on CPU, sequential
    static let zeroGPU = PipelineConfig(
        yoloPath: "yolo11s-seg.mlpackage", posePath: "dwpose_m.mlpackage",
        yoloUnits: .cpuAndNeuralEngine, poseUnits: .cpuOnly,
        poseH: 256, poseW: 192, parallel: false)

    /// Max accuracy: large model, both GPU, sequential
    static let maxAccuracy = PipelineConfig(
        yoloPath: "yolo11s-seg.mlpackage", posePath: "rtmpose.mlpackage",
        yoloUnits: .cpuAndGPU, poseUnits: .cpuAndGPU,
        poseH: 384, poseW: 288, parallel: false)

    /// GPU-light large model: YOLO on ANE, RTMPose on GPU, sequential
    static let largeGPULight = PipelineConfig(
        yoloPath: "yolo11s-seg.mlpackage", posePath: "rtmpose.mlpackage",
        yoloUnits: .all, poseUnits: .all,
        poseH: 384, poseW: 288, parallel: false)
}

// *** CHANGE THIS TO SWAP CONFIGS ***
let ACTIVE_CONFIG = PipelineConfig.parallelANEGPU

// ============================================================================
// MARK: - Frame Result
// ============================================================================

struct FrameResult {
    let contour: [(Float, Float)]
    let keypoints: [(Float, Float)]
    let confidence: [Float]
    let bbox: (Float, Float, Float, Float)
    let mask: MaskData?
    let frameTimeMs: Double
}

// ============================================================================
// MARK: - Pipeline Engine
// ============================================================================

@MainActor
class PipelineEngine: NSObject, ObservableObject {
    @Published var latestResult: FrameResult?
    @Published var fps: Double = 0
    @Published var isRunning = false
    @Published var imageSize: CGSize = CGSize(width: 1920, height: 1080)

    private var yoloModel: MLModel?
    private var poseModel: MLModel?
    private var yoloInName = ""
    private var yoloOutNames: [String] = []
    private var poseInName = ""
    private var poseOutNames: [String] = []
    private var yoloPB: CVPixelBuffer?

    private var prevBbox: (Float, Float, Float, Float)?
    private let config: PipelineConfig

    private var captureSession: AVCaptureSession?
    private let videoQueue = DispatchQueue(label: "video-capture")

    // Timing
    private var frameTimes: [Double] = []
    private let maxFrameTimes = 60

    init(config: PipelineConfig) {
        self.config = config
        super.init()
    }

    func start(modelDir: String) async throws {
        // Load models
        let yoloURL = URL(fileURLWithPath: "\(modelDir)/\(config.yoloPath)")
        let poseURL = URL(fileURLWithPath: "\(modelDir)/\(config.posePath)")

        let yoloCompiled = try await MLModel.compileModel(at: yoloURL)
        let poseCompiled = try await MLModel.compileModel(at: poseURL)

        let yCfg = MLModelConfiguration()
        yCfg.computeUnits = config.yoloUnits
        yoloModel = try MLModel(contentsOf: yoloCompiled, configuration: yCfg)

        let pCfg = MLModelConfiguration()
        pCfg.computeUnits = config.poseUnits
        poseModel = try MLModel(contentsOf: poseCompiled, configuration: pCfg)

        yoloInName = yoloModel!.modelDescription.inputDescriptionsByName.keys.first!
        yoloOutNames = Array(yoloModel!.modelDescription.outputDescriptionsByName.keys).sorted()
        poseInName = poseModel!.modelDescription.inputDescriptionsByName.keys.first!
        poseOutNames = Array(poseModel!.modelDescription.outputDescriptionsByName.keys).sorted()

        yoloPB = makePixelBuffer(YOLO_SIZE, YOLO_SIZE)

        // Start camera
        try startCamera()
        isRunning = true
    }

    private func startCamera() throws {
        let session = AVCaptureSession()
        session.sessionPreset = .hd1920x1080

        guard let device = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device)
        else { throw NSError(domain: "Camera", code: 1, userInfo: [NSLocalizedDescriptionKey: "No camera"]) }

        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: videoQueue)
        session.addOutput(output)

        captureSession = session
        session.startRunning()
    }

    func stop() {
        captureSession?.stopRunning()
        isRunning = false
    }

    // Process a single frame
    private func processFrame(_ pixelBuffer: CVPixelBuffer) async {
        let t0 = CFAbsoluteTimeGetCurrent()
        let imgW = CVPixelBufferGetWidth(pixelBuffer)
        let imgH = CVPixelBufferGetHeight(pixelBuffer)
        imageSize = CGSize(width: imgW, height: imgH)

        guard let yolo = yoloModel, let pose = poseModel, let yoloPB = yoloPB else { return }

        if config.parallel {
            await processParallel(pixelBuffer, imgW: imgW, imgH: imgH,
                                   yolo: yolo, pose: pose, yoloPB: yoloPB, t0: t0)
        } else {
            await processSequential(pixelBuffer, imgW: imgW, imgH: imgH,
                                     yolo: yolo, pose: pose, yoloPB: yoloPB, t0: t0)
        }
    }

    private func processParallel(_ pb: CVPixelBuffer, imgW: Int, imgH: Int,
                                  yolo: MLModel, pose: MLModel, yoloPB: CVPixelBuffer,
                                  t0: Double) async {
        let currentBbox = prevBbox

        async let yoloTask: (Detection?, [(Float, Float)]?, MaskData?) = {
            let (scale, padL, padT) = letterboxPixelBuffer(src: pb, into: yoloPB,
                                                            imgW: imgW, imgH: imgH)
            let input = try! await MLDictionaryFeatureProvider(dictionary: [
                self.yoloInName: MLFeatureValue(pixelBuffer: yoloPB)
            ])
            let pred = try! await yolo.prediction(from: input)
            let det = await parseDetections(pred, outputNames: self.yoloOutNames,
                                      imgW: imgW, imgH: imgH,
                                      lbScale: scale, padLeft: padL, padTop: padT)
            var contour: [(Float, Float)]? = nil
            var maskData: MaskData? = nil
            if let d = det {
                contour = decodeContour(d, imgW: imgW, imgH: imgH,
                                        lbScale: scale, padLeft: padL, padTop: padT)
                maskData = decodeMask(d, imgW: imgW, imgH: imgH,
                                      lbScale: scale, padLeft: padL, padTop: padT)
            }
            return (det, contour, maskData)
        }()

        async let poseTask: PoseResult? = {
            guard let bbox = currentBbox else { return nil }
            return try? await runPoseOnPixelBuffer(
                model: pose, inputName: self.poseInName, outputNames: self.poseOutNames,
                pixelBuffer: pb, bbox: bbox,
                poseH: self.config.poseH, poseW: self.config.poseW)
        }()

        let (yResult, pResult) = await (yoloTask, poseTask)
        let (det, contour, maskData) = yResult

        if let d = det { prevBbox = d.bbox }

        let tEnd = CFAbsoluteTimeGetCurrent()
        let frameMs = (tEnd - t0) * 1000

        updateResult(contour: contour, pose: pResult, mask: maskData,
                     bbox: det?.bbox ?? prevBbox, frameMs: frameMs)
    }

    private func processSequential(_ pb: CVPixelBuffer, imgW: Int, imgH: Int,
                                    yolo: MLModel, pose: MLModel, yoloPB: CVPixelBuffer,
                                    t0: Double) async {
        let (scale, padL, padT) = letterboxPixelBuffer(src: pb, into: yoloPB,
                                                        imgW: imgW, imgH: imgH)
        let yoloInput = try! await MLDictionaryFeatureProvider(dictionary: [
            yoloInName: MLFeatureValue(pixelBuffer: yoloPB)
        ])
        let yoloPred = try! await yolo.prediction(from: yoloInput)
        let det = parseDetections(yoloPred, outputNames: yoloOutNames,
                                  imgW: imgW, imgH: imgH,
                                  lbScale: scale, padLeft: padL, padTop: padT)

        var contour: [(Float, Float)]? = nil
        var maskData: MaskData? = nil
        if let d = det {
            contour = decodeContour(d, imgW: imgW, imgH: imgH,
                                    lbScale: scale, padLeft: padL, padTop: padT)
            maskData = decodeMask(d, imgW: imgW, imgH: imgH,
                                  lbScale: scale, padLeft: padL, padTop: padT)
        }

        guard let d = det else {
            let tEnd = CFAbsoluteTimeGetCurrent()
            updateResult(contour: nil, pose: nil, mask: nil, bbox: nil, frameMs: (tEnd - t0) * 1000)
            return
        }

        let pResult = try? await runPoseOnPixelBuffer(
            model: pose, inputName: poseInName, outputNames: poseOutNames,
            pixelBuffer: pb, bbox: d.bbox,
            poseH: config.poseH, poseW: config.poseW)

        let tEnd = CFAbsoluteTimeGetCurrent()
        updateResult(contour: contour, pose: pResult, mask: maskData,
                     bbox: d.bbox, frameMs: (tEnd - t0) * 1000)
    }

    private func updateResult(contour: [(Float, Float)]?, pose: PoseResult?,
                               mask: MaskData?,
                               bbox: (Float, Float, Float, Float)?, frameMs: Double) {
        frameTimes.append(frameMs)
        if frameTimes.count > maxFrameTimes { frameTimes.removeFirst() }
        let avgFps = 1000.0 / (frameTimes.reduce(0, +) / Double(frameTimes.count))

        let result = FrameResult(
            contour: contour ?? [],
            keypoints: pose?.keypoints ?? [],
            confidence: pose?.confidence ?? [],
            bbox: bbox ?? (0, 0, 0, 0),
            mask: mask,
            frameTimeMs: frameMs
        )

        Task { @MainActor in
            self.latestResult = result
            self.fps = avgFps
        }
    }
}

// MARK: - Camera Delegate

extension PipelineEngine: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput,
                                    didOutput sampleBuffer: CMSampleBuffer,
                                    from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        Task { await self.processFrame(pb) }
    }
}

// ============================================================================
// MARK: - Visualization View
// ============================================================================

struct OverlayView: View {
    let result: FrameResult?
    let viewSize: CGSize
    let imageSize: CGSize
    let mirrored: Bool  // true for front camera

    /// Map image-space X to view-space X, with optional mirror.
    func mapX(_ x: Double, _ scaleX: Double, _ width: Double) -> Double {
        let mapped = x * scaleX
        return mirrored ? (width - mapped) : mapped
    }

    func maskToCGImage(_ mask: MaskData) -> CGImage? {
        var rgba = [UInt8](repeating: 0, count: mask.width * mask.height * 4)
        for i in 0..<(mask.width * mask.height) {
            let a = UInt8(min(255, max(0, mask.values[i] * 160)))
            rgba[i * 4] = 0       // R (premultiplied)
            rgba[i * 4 + 1] = a   // G (premultiplied: 255 * a/255 = a)
            rgba[i * 4 + 2] = 0   // B (premultiplied)
            rgba[i * 4 + 3] = a   // A
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        return CGImage(width: mask.width, height: mask.height,
                       bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: mask.width * 4,
                       space: colorSpace,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                       provider: provider,
                       decode: nil, shouldInterpolate: true,
                       intent: .defaultIntent)
    }

    var body: some View {
        Canvas { context, size in
            guard let r = result else { return }

            let scaleX = size.width / imageSize.width
            let scaleY = size.height / imageSize.height

            // Draw raw mask as bottom layer
            if let mask = r.mask, let cgImage = maskToCGImage(mask) {
                let img = Image(decorative: cgImage, scale: 1)
                if mirrored {
                    var ctx = context
                    ctx.concatenate(CGAffineTransform(scaleX: -1, y: 1)
                        .translatedBy(x: -size.width, y: 0))
                    ctx.draw(img, in: CGRect(origin: .zero, size: size))
                } else {
                    context.draw(img, in: CGRect(origin: .zero, size: size))
                }
            }

            // Draw contour
            if r.contour.count > 2 {
                var path = Path()
                path.move(to: CGPoint(x: mapX(Double(r.contour[0].0), scaleX, size.width),
                                      y: Double(r.contour[0].1) * scaleY))
                for pt in r.contour.dropFirst() {
                    path.addLine(to: CGPoint(x: mapX(Double(pt.0), scaleX, size.width),
                                             y: Double(pt.1) * scaleY))
                }
                path.closeSubpath()
                context.stroke(path, with: .color(.green), lineWidth: 2)
            }

            // Draw bbox
            let (bx1, by1, bx2, by2) = r.bbox
            if bx2 > bx1 {
                let left = mapX(Double(bx1), scaleX, size.width)
                let right = mapX(Double(bx2), scaleX, size.width)
                let bboxRect = CGRect(
                    x: min(left, right), y: Double(by1) * scaleY,
                    width: abs(right - left), height: Double(by2 - by1) * scaleY)
                context.stroke(Path(bboxRect), with: .color(.yellow), lineWidth: 1)
            }

            // Draw keypoints
            for (i, kpt) in r.keypoints.enumerated() {
                let conf = i < r.confidence.count ? r.confidence[i] : 0
                if conf < 0.3 { continue }

                let x = mapX(Double(kpt.0), scaleX, size.width)
                let y = Double(kpt.1) * scaleY
                let radius: Double = 3

                let color: Color
                if i < 17 { color = .red }
                else if i < 23 { color = .orange }
                else if i < 91 { color = .blue }
                else { color = .cyan }

                let circle = Path(ellipseIn: CGRect(x: x - radius, y: y - radius,
                                                     width: radius * 2, height: radius * 2))
                context.fill(circle, with: .color(color))
            }
        }
    }
}

// ============================================================================
// MARK: - Main App
// ============================================================================

struct CameraView: View {
    @StateObject private var engine = PipelineEngine(config: ACTIVE_CONFIG)

    var body: some View {
        ZStack {
            Color.black.edgesIgnoringSafeArea(.all)

            // Overlay
            OverlayView(result: engine.latestResult,
                        viewSize: CGSize(width: 960, height: 540),
                        imageSize: engine.imageSize,
                        mirrored: true)
                .frame(width: 960, height: 540)

            // FPS counter
            VStack {
                HStack {
                    Spacer()
                    VStack(alignment: .trailing) {
                        Text(String(format: "%.1f FPS", engine.fps))
                            .font(.system(.title2, design: .monospaced))
                            .foregroundColor(.white)
                        if let r = engine.latestResult {
                            Text(String(format: "%.1f ms", r.frameTimeMs))
                                .font(.system(.caption, design: .monospaced))
                                .foregroundColor(.gray)
                            Text("\(r.keypoints.count) kpts, \(r.contour.count) contour pts")
                                .font(.system(.caption, design: .monospaced))
                                .foregroundColor(.gray)
                        }
                    }
                    .padding()
                    .background(Color.black.opacity(0.6))
                    .cornerRadius(8)
                }
                Spacer()
            }
            .padding()
        }
        .task {
            do {
                try await engine.start(modelDir: FileManager.default.currentDirectoryPath)
            } catch {
                print("Failed to start: \(error)")
            }
        }
    }
}

@main
struct CameraTestApp: App {
    var body: some Scene {
        WindowGroup {
            CameraView()
                .frame(width: 960, height: 540)
        }
    }
}
