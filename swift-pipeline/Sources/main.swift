import Foundation
import CoreML
import CoreVideo
import Accelerate

// ============================================================================
// MARK: - Constants
// ============================================================================

let YOLO_SIZE = 640
let YOLO_CONF_THRESH: Float = 0.25
let PERSON_CLASS = 0
let RTM_H = 384
let RTM_W = 288
let SIMCC_SPLIT: Float = 2.0
let FRAME_W = 1920
let FRAME_H = 1080
let FRAME_BYTES = FRAME_W * FRAME_H * 4 // BGRA

// RTMPose normalization (BGR order)
let MEAN: [Float] = [123.675, 116.28, 103.53]
let STD: [Float]  = [58.395,  57.12,  57.375]

// Precomputed letterbox params (constant for 1920x1080 → 640x640)
let LB_SCALE = min(Float(YOLO_SIZE) / Float(FRAME_H), Float(YOLO_SIZE) / Float(FRAME_W))
let LB_NEW_W = Int(Float(FRAME_W) * LB_SCALE)
let LB_NEW_H = Int(Float(FRAME_H) * LB_SCALE)
let LB_PAD_LEFT = (YOLO_SIZE - LB_NEW_W) / 2
let LB_PAD_TOP  = (YOLO_SIZE - LB_NEW_H) / 2

// Proto-space (160 = 640/4)
let PROTO_SCALE = Float(160) / Float(YOLO_SIZE)
let PROTO_PAD_L = Int(Float(LB_PAD_LEFT) * PROTO_SCALE)
let PROTO_PAD_T = Int(Float(LB_PAD_TOP) * PROTO_SCALE)
let PROTO_CW    = Int(Float(LB_NEW_W) * PROTO_SCALE)
let PROTO_CH    = Int(Float(LB_NEW_H) * PROTO_SCALE)
let PROTO_TO_IMG_X = Float(FRAME_W) / Float(PROTO_CW)
let PROTO_TO_IMG_Y = Float(FRAME_H) / Float(PROTO_CH)

// ============================================================================
// MARK: - Frame Reader (ffmpeg pipe → raw BGRA, simulates camera)
// ============================================================================

class FrameReader {
    private let process = Process()
    private let pipe = Pipe()
    private let handle: FileHandle

    init(videoPath: String) {
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            "ffmpeg", "-i", videoPath,
            "-vf", "scale=\(FRAME_W):\(FRAME_H)",
            "-f", "rawvideo", "-pix_fmt", "bgra",
            "-v", "quiet", "-"
        ]
        process.standardOutput = pipe
        handle = pipe.fileHandleForReading
    }

    func start() throws { try process.run() }

    func readFrame() -> Data? {
        let data = handle.readData(ofLength: FRAME_BYTES)
        return data.count == FRAME_BYTES ? data : nil
    }

    func stop() { process.terminate() }
}

// ============================================================================
// MARK: - CVPixelBuffer Helpers
// ============================================================================

func makePixelBuffer(_ w: Int, _ h: Int) -> CVPixelBuffer {
    var pb: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, w, h,
                        kCVPixelFormatType_32BGRA,
                        [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary,
                        &pb)
    return pb!
}

/// Letterbox a raw BGRA frame into a 640×640 CVPixelBuffer using vImage.
func letterbox(frameData: Data, into dest: CVPixelBuffer) {
    CVPixelBufferLockBaseAddress(dest, [])
    let base = CVPixelBufferGetBaseAddress(dest)!
    let rowBytes = CVPixelBufferGetBytesPerRow(dest)

    // Fill with grey (114,114,114,255)
    var pattern: UInt32 = 0
    withUnsafeMutableBytes(of: &pattern) { p in
        p[0] = 114; p[1] = 114; p[2] = 114; p[3] = 255
    }
    memset_pattern4(base, &pattern, rowBytes * YOLO_SIZE)

    // Resize source into the content region of the canvas
    frameData.withUnsafeBytes { src in
        var srcBuf = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: src.baseAddress!),
            height: vImagePixelCount(FRAME_H),
            width: vImagePixelCount(FRAME_W),
            rowBytes: FRAME_W * 4
        )
        var dstBuf = vImage_Buffer(
            data: base.advanced(by: LB_PAD_TOP * rowBytes + LB_PAD_LEFT * 4),
            height: vImagePixelCount(LB_NEW_H),
            width: vImagePixelCount(LB_NEW_W),
            rowBytes: rowBytes
        )
        vImageScale_ARGB8888(&srcBuf, &dstBuf, nil, vImage_Flags(kvImageNoFlags))
    }
    CVPixelBufferUnlockBaseAddress(dest, [])
}

// ============================================================================
// MARK: - YOLO Processing
// ============================================================================

struct Detection {
    let bbox: (Float, Float, Float, Float) // x1,y1,x2,y2 image coords
    let confidence: Float
    let coeffs: [Float]                    // 32 mask coefficients
    let proto: MLMultiArray               // (1,32,160,160) prototype tensor
}

func parseDetections(_ prediction: MLFeatureProvider,
                     outputNames: [String]) -> Detection? {
    var detArr: MLMultiArray?
    var protoArr: MLMultiArray?
    for name in outputNames {
        guard let a = prediction.featureValue(for: name)?.multiArrayValue else { continue }
        if a.shape.count == 3 { detArr = a }
        else if a.shape.count == 4 { protoArr = a }
    }
    guard let det = detArr, let proto = protoArr else { return nil }

    let nAnchors = det.shape[2].intValue
    let ptr = det.dataPointer.bindMemory(to: Float.self, capacity: det.count)

    // Find best person detection
    let personOff = (4 + PERSON_CLASS) * nAnchors
    var bestIdx = -1
    var bestScore = YOLO_CONF_THRESH
    for i in 0..<nAnchors {
        let s = ptr[personOff + i]
        if s > bestScore { bestScore = s; bestIdx = i }
    }
    guard bestIdx >= 0 else { return nil }

    // xywh → xyxy in image coords
    let cx = ptr[0 * nAnchors + bestIdx]
    let cy = ptr[1 * nAnchors + bestIdx]
    let w  = ptr[2 * nAnchors + bestIdx]
    let h  = ptr[3 * nAnchors + bestIdx]
    let x1 = max(0, (cx - w/2 - Float(LB_PAD_LEFT)) / LB_SCALE)
    let y1 = max(0, (cy - h/2 - Float(LB_PAD_TOP))  / LB_SCALE)
    let x2 = min(Float(FRAME_W), (cx + w/2 - Float(LB_PAD_LEFT)) / LB_SCALE)
    let y2 = min(Float(FRAME_H), (cy + h/2 - Float(LB_PAD_TOP))  / LB_SCALE)

    var coeffs = [Float](repeating: 0, count: 32)
    for c in 0..<32 { coeffs[c] = ptr[(84 + c) * nAnchors + bestIdx] }

    return Detection(bbox: (x1, y1, x2, y2), confidence: bestScore,
                     coeffs: coeffs, proto: proto)
}

// ============================================================================
// MARK: - Contour Extraction (at proto resolution)
// ============================================================================

func decodeContour(_ det: Detection) -> [(Float, Float)] {
    let proto = det.proto
    let pPtr = proto.dataPointer.bindMemory(to: Float.self, capacity: proto.count)
    let maskSize = 160 * 160

    // Weighted sum: mask = coeffs @ proto channels
    var mask = [Float](repeating: 0, count: maskSize)
    for c in 0..<32 {
        var coeff = det.coeffs[c]
        // mask[i] += coeff * proto[c, i]
        vDSP_vsma(pPtr.advanced(by: c * maskSize), 1,
                  &coeff, mask, 1, &mask, 1, vDSP_Length(maskSize))
    }

    // Sigmoid in-place
    for i in 0..<maskSize { mask[i] = 1.0 / (1.0 + exp(-mask[i])) }

    // Extract content region, threshold
    var binary = [UInt8](repeating: 0, count: PROTO_CW * PROTO_CH)
    for y in 0..<PROTO_CH {
        for x in 0..<PROTO_CW {
            if mask[(y + PROTO_PAD_T) * 160 + (x + PROTO_PAD_L)] > 0.5 {
                binary[y * PROTO_CW + x] = 1
            }
        }
    }

    // Moore boundary tracing
    let raw = traceContour(binary, PROTO_CW, PROTO_CH)
    return raw.map { (Float($0) * PROTO_TO_IMG_X, Float($1) * PROTO_TO_IMG_Y) }
}

func traceContour(_ mask: [UInt8], _ w: Int, _ h: Int) -> [(Int, Int)] {
    let dx = [1, 1, 0, -1, -1, -1, 0, 1]
    let dy = [0, 1, 1,  1,  0, -1, -1, -1]

    // Find start pixel
    var sx = -1, sy = -1
    for y in 0..<h {
        for x in 0..<w {
            if mask[y * w + x] != 0 { sx = x; sy = y; break }
        }
        if sx >= 0 { break }
    }
    guard sx >= 0 else { return [] }

    var contour: [(Int, Int)] = [(sx, sy)]
    var x = sx, y = sy, back = 4
    for _ in 0..<(w * h * 2) {
        let start = (back + 1) % 8
        var moved = false
        for i in 0..<8 {
            let d = (start + i) % 8
            let nx = x + dx[d], ny = y + dy[d]
            if nx >= 0, nx < w, ny >= 0, ny < h, mask[ny * w + nx] != 0 {
                x = nx; y = ny; back = (d + 4) % 8
                if x == sx && y == sy { return contour }
                contour.append((x, y)); moved = true; break
            }
        }
        if !moved { break }
    }
    return contour
}

// ============================================================================
// MARK: - RTMPose Processing
// ============================================================================

struct PoseResult {
    let keypoints: [(Float, Float)]  // 133 points in image coords
    let confidence: [Float]
}

func runPose(model: MLModel, inputName: String, outputNames: [String],
             frameData: Data, bbox: (Float, Float, Float, Float)) throws -> PoseResult? {
    let (bx1, by1, bx2, by2) = bbox
    let cx = (bx1 + bx2) / 2, cy = (by1 + by2) / 2
    let bw = bx2 - bx1, bh = by2 - by1
    let cropScale = max(bw / Float(RTM_W), bh / Float(RTM_H)) * 1.25
    let nw = Float(RTM_W) * cropScale, nh = Float(RTM_H) * cropScale

    let x1c = max(0, Int(cx - nw / 2))
    let y1c = max(0, Int(cy - nh / 2))
    let x2c = min(FRAME_W, Int(cx + nw / 2))
    let y2c = min(FRAME_H, Int(cy + nh / 2))
    let cropW = x2c - x1c, cropH = y2c - y1c
    guard cropW > 0, cropH > 0 else { return nil }

    // Crop + resize using vImage
    let resized: [UInt8] = frameData.withUnsafeBytes { src in
        var cropBuf = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: src.baseAddress!
                    .advanced(by: y1c * FRAME_W * 4 + x1c * 4)),
            height: vImagePixelCount(cropH), width: vImagePixelCount(cropW),
            rowBytes: FRAME_W * 4
        )
        var out = [UInt8](repeating: 0, count: RTM_W * RTM_H * 4)
        out.withUnsafeMutableBytes { dst in
            var dstBuf = vImage_Buffer(
                data: dst.baseAddress!, height: vImagePixelCount(RTM_H),
                width: vImagePixelCount(RTM_W), rowBytes: RTM_W * 4
            )
            vImageScale_ARGB8888(&cropBuf, &dstBuf, nil, vImage_Flags(kvImageNoFlags))
        }
        return out
    }

    // Normalize BGRA → CHW float32, fill MLMultiArray
    let pixels = RTM_W * RTM_H
    let arr = try MLMultiArray(shape: [1, 3, NSNumber(value: RTM_H), NSNumber(value: RTM_W)],
                               dataType: .float32)
    let fp = arr.dataPointer.bindMemory(to: Float.self, capacity: 3 * pixels)
    for i in 0..<pixels {
        fp[i]              = (Float(resized[i * 4])     - MEAN[0]) / STD[0] // B
        fp[pixels + i]     = (Float(resized[i * 4 + 1]) - MEAN[1]) / STD[1] // G
        fp[2 * pixels + i] = (Float(resized[i * 4 + 2]) - MEAN[2]) / STD[2] // R
    }

    // Infer
    let input = try MLDictionaryFeatureProvider(dictionary: [
        inputName: MLFeatureValue(multiArray: arr)
    ])
    let pred = try model.prediction(from: input)

    // Decode SimCC
    guard let sxArr = pred.featureValue(for: outputNames[0])?.multiArrayValue,
          let syArr = pred.featureValue(for: outputNames[1])?.multiArrayValue
    else { return nil }

    let nKpts = sxArr.shape[1].intValue
    let xLen = sxArr.shape[2].intValue
    let yLen = syArr.shape[2].intValue
    let xp = sxArr.dataPointer.bindMemory(to: Float.self, capacity: sxArr.count)
    let yp = syArr.dataPointer.bindMemory(to: Float.self, capacity: syArr.count)

    var kpts = [(Float, Float)]()
    var conf = [Float]()
    kpts.reserveCapacity(nKpts)
    conf.reserveCapacity(nKpts)

    for k in 0..<nKpts {
        var maxXV: Float = 0, maxXI: vDSP_Length = 0
        var maxYV: Float = 0, maxYI: vDSP_Length = 0
        vDSP_maxvi(xp.advanced(by: k * xLen), 1, &maxXV, &maxXI, vDSP_Length(xLen))
        vDSP_maxvi(yp.advanced(by: k * yLen), 1, &maxYV, &maxYI, vDSP_Length(yLen))

        let px = Float(maxXI) / SIMCC_SPLIT * cropScale + Float(x1c)
        let py = Float(maxYI) / SIMCC_SPLIT * cropScale + Float(y1c)
        kpts.append((px, py))
        conf.append(min(maxXV, maxYV))
    }
    return PoseResult(keypoints: kpts, confidence: conf)
}

// ============================================================================
// MARK: - Timing Helpers
// ============================================================================

func median(_ a: [Double]) -> Double {
    let s = a.sorted()
    return s.isEmpty ? 0 : (s.count % 2 == 0 ? (s[s.count/2-1]+s[s.count/2])/2 : s[s.count/2])
}
func pct95(_ a: [Double]) -> Double {
    let s = a.sorted()
    return s.isEmpty ? 0 : s[min(Int(Double(s.count-1)*0.95), s.count-1)]
}

// ============================================================================
// MARK: - Async RTMPose
// ============================================================================

func runPoseAsync(model: MLModel, inputName: String, outputNames: [String],
                  frameData: Data, bbox: (Float, Float, Float, Float)) async throws -> PoseResult? {
    let (bx1, by1, bx2, by2) = bbox
    let cx = (bx1 + bx2) / 2, cy = (by1 + by2) / 2
    let bw = bx2 - bx1, bh = by2 - by1
    let cropScale = max(bw / Float(RTM_W), bh / Float(RTM_H)) * 1.25
    let nw = Float(RTM_W) * cropScale, nh = Float(RTM_H) * cropScale

    let x1c = max(0, Int(cx - nw / 2))
    let y1c = max(0, Int(cy - nh / 2))
    let x2c = min(FRAME_W, Int(cx + nw / 2))
    let y2c = min(FRAME_H, Int(cy + nh / 2))
    let cropW = x2c - x1c, cropH = y2c - y1c
    guard cropW > 0, cropH > 0 else { return nil }

    let resized: [UInt8] = frameData.withUnsafeBytes { src in
        var cropBuf = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: src.baseAddress!
                    .advanced(by: y1c * FRAME_W * 4 + x1c * 4)),
            height: vImagePixelCount(cropH), width: vImagePixelCount(cropW),
            rowBytes: FRAME_W * 4
        )
        var out = [UInt8](repeating: 0, count: RTM_W * RTM_H * 4)
        out.withUnsafeMutableBytes { dst in
            var dstBuf = vImage_Buffer(
                data: dst.baseAddress!, height: vImagePixelCount(RTM_H),
                width: vImagePixelCount(RTM_W), rowBytes: RTM_W * 4
            )
            vImageScale_ARGB8888(&cropBuf, &dstBuf, nil, vImage_Flags(kvImageNoFlags))
        }
        return out
    }

    let pixels = RTM_W * RTM_H
    let arr = try MLMultiArray(shape: [1, 3, NSNumber(value: RTM_H), NSNumber(value: RTM_W)],
                               dataType: .float32)
    let fp = arr.dataPointer.bindMemory(to: Float.self, capacity: 3 * pixels)
    for i in 0..<pixels {
        fp[i]              = (Float(resized[i * 4])     - MEAN[0]) / STD[0]
        fp[pixels + i]     = (Float(resized[i * 4 + 1]) - MEAN[1]) / STD[1]
        fp[2 * pixels + i] = (Float(resized[i * 4 + 2]) - MEAN[2]) / STD[2]
    }

    let input = try MLDictionaryFeatureProvider(dictionary: [
        inputName: MLFeatureValue(multiArray: arr)
    ])
    let pred = try await model.prediction(from: input)

    guard let sxArr = pred.featureValue(for: outputNames[0])?.multiArrayValue,
          let syArr = pred.featureValue(for: outputNames[1])?.multiArrayValue
    else { return nil }

    let nKpts = sxArr.shape[1].intValue
    let xLen = sxArr.shape[2].intValue
    let yLen = syArr.shape[2].intValue
    let xp = sxArr.dataPointer.bindMemory(to: Float.self, capacity: sxArr.count)
    let yp = syArr.dataPointer.bindMemory(to: Float.self, capacity: syArr.count)

    var kpts = [(Float, Float)]()
    var conf = [Float]()
    kpts.reserveCapacity(nKpts)
    conf.reserveCapacity(nKpts)

    for k in 0..<nKpts {
        var maxXV: Float = 0, maxXI: vDSP_Length = 0
        var maxYV: Float = 0, maxYI: vDSP_Length = 0
        vDSP_maxvi(xp.advanced(by: k * xLen), 1, &maxXV, &maxXI, vDSP_Length(xLen))
        vDSP_maxvi(yp.advanced(by: k * yLen), 1, &maxYV, &maxYI, vDSP_Length(yLen))

        let px = Float(maxXI) / SIMCC_SPLIT * cropScale + Float(x1c)
        let py = Float(maxYI) / SIMCC_SPLIT * cropScale + Float(y1c)
        kpts.append((px, py))
        conf.append(min(maxXV, maxYV))
    }
    return PoseResult(keypoints: kpts, confidence: conf)
}

// ============================================================================
// MARK: - Main (async entry point)
// ============================================================================

func runPipeline() async throws {
    let args = CommandLine.arguments

    if args.count >= 2 && args[1] == "--bench" {
        try await runStandaloneBench()
        return
    }

    guard args.count >= 2 else {
        print("Usage: CoreMLPipeline <video-path> [num-frames]")
        print("       CoreMLPipeline --bench")
        return
    }
    let videoPath = args[1]
    let numFrames = args.count >= 3 ? (Int(args[2]) ?? 200) : 200
    let modelDir = FileManager.default.currentDirectoryPath

    // Load and compile models
    print("Compiling models...")
    let yoloURL = URL(fileURLWithPath: "\(modelDir)/yolo11s-seg.mlpackage")
    let poseURL = URL(fileURLWithPath: "\(modelDir)/rtmpose.mlpackage")

    let yoloCompiled = try await MLModel.compileModel(at: yoloURL)
    let poseCompiled = try await MLModel.compileModel(at: poseURL)

        let config = MLModelConfiguration()
        config.computeUnits = .all
        let yoloModel = try MLModel(contentsOf: yoloCompiled, configuration: config)
        let poseModel = try MLModel(contentsOf: poseCompiled, configuration: config)

        let yoloOutNames = Array(yoloModel.modelDescription.outputDescriptionsByName.keys).sorted()
        let yoloInName   = yoloModel.modelDescription.inputDescriptionsByName.keys.first!
        let poseOutNames = Array(poseModel.modelDescription.outputDescriptionsByName.keys).sorted()
        let poseInName   = poseModel.modelDescription.inputDescriptionsByName.keys.first!

        print("  YOLO input: \(yoloInName), outputs: \(yoloOutNames)")
        print("  Pose input: \(poseInName), outputs: \(poseOutNames)")

        // Preallocate YOLO letterbox buffer
        let yoloPB = makePixelBuffer(YOLO_SIZE, YOLO_SIZE)

        // Start ffmpeg reader
        print("Starting ffmpeg (\(FRAME_W)x\(FRAME_H) BGRA)...")
        let reader = FrameReader(videoPath: videoPath)
        try reader.start()

        // --- Pipelined frame reader (1-frame overlap) ---
        var pendingFrame: Data?
        let frameLock = NSLock()
        let frameReady = DispatchSemaphore(value: 0)
        let readQueue = DispatchQueue(label: "reader", qos: .userInitiated)

        func kickRead() {
            readQueue.async {
                let d = reader.readFrame()
                frameLock.lock()
                pendingFrame = d
                frameLock.unlock()
                frameReady.signal()
            }
        }

        func takeFrame() -> Data? {
            frameReady.wait()
            frameLock.lock()
            let d = pendingFrame
            pendingFrame = nil
            frameLock.unlock()
            return d
        }

        // Warmup
        print("Warming up (5 frames)...")
        kickRead()
        for _ in 0..<5 {
            guard let fd = takeFrame() else { break }
            kickRead()
            letterbox(frameData: fd, into: yoloPB)
            let inp = try MLDictionaryFeatureProvider(dictionary: [
                yoloInName: MLFeatureValue(pixelBuffer: yoloPB)
            ])
            let _ = try await yoloModel.prediction(from: inp)
        }

        // --- Main processing loop ---
        print("Processing \(numFrames) frames...\n")

        var t_wait  = [Double]()
        var t_ypre  = [Double]()
        var t_yinf  = [Double]()
        var t_ypost = [Double]()
        var t_cont  = [Double]()
        var t_pose  = [Double]()
        var t_total = [Double]()

        kickRead()

        for i in 0..<numFrames {
            let t0 = CFAbsoluteTimeGetCurrent()
            guard let frameData = takeFrame() else {
                print("End of video at frame \(i)")
                break
            }
            let tw = CFAbsoluteTimeGetCurrent()

            kickRead()

            // YOLO preprocess
            letterbox(frameData: frameData, into: yoloPB)
            let t1 = CFAbsoluteTimeGetCurrent()

            // YOLO inference (async)
            let yoloInput = try MLDictionaryFeatureProvider(dictionary: [
                yoloInName: MLFeatureValue(pixelBuffer: yoloPB)
            ])
            let yoloPred = try await yoloModel.prediction(from: yoloInput)
            let t2 = CFAbsoluteTimeGetCurrent()

            // YOLO postprocess
            let detection = parseDetections(yoloPred, outputNames: yoloOutNames)
            let t3 = CFAbsoluteTimeGetCurrent()

            guard let det = detection else {
                t_wait.append(tw - t0); t_ypre.append(t1 - tw)
                t_yinf.append(t2 - t1); t_ypost.append(t3 - t2)
                t_cont.append(0); t_pose.append(0)
                t_total.append(t3 - tw)
                continue
            }

            // Contour at proto resolution
            let _ = decodeContour(det)
            let t4 = CFAbsoluteTimeGetCurrent()

            // RTMPose (async preprocess + infer + decode)
            let _ = try await runPoseAsync(model: poseModel, inputName: poseInName,
                                           outputNames: poseOutNames,
                                           frameData: frameData, bbox: det.bbox)
            let t5 = CFAbsoluteTimeGetCurrent()

            t_wait.append(tw - t0)
            t_ypre.append(t1 - tw)
            t_yinf.append(t2 - t1)
            t_ypost.append(t3 - t2)
            t_cont.append(t4 - t3)
            t_pose.append(t5 - t4)
            t_total.append(t5 - tw)

            if (i + 1) % 50 == 0 {
                let m = median(t_total) * 1000
                print("  Frame \(i+1)/\(numFrames) — median \(String(format: "%.1f", m))ms " +
                      "(\(String(format: "%.1f", 1000/m)) FPS)")
            }
        }

        reader.stop()

        // ============================================================================
        // Report
        // ============================================================================
        print("\n" + String(repeating: "=", count: 70))
        print("Swift CoreML Pipeline — async predictions, 1-frame pipelined read")
        print(String(repeating: "=", count: 70))

        let totalMed = median(t_total) * 1000
        let stages: [(String, [Double])] = [
            ("Frame wait (overlap)",  t_wait),
            ("YOLO preprocess",       t_ypre),
            ("YOLO inference",        t_yinf),
            ("YOLO postprocess",      t_ypost),
            ("Contour (160px)",       t_cont),
            ("RTMPose total",         t_pose),
        ]

        for (label, arr) in stages {
            let m = median(arr) * 1000
            let pct = totalMed > 0 ? m / totalMed * 100 : 0
            let bar = String(repeating: "#", count: max(0, Int(pct / 2)))
            print("  \(label.padding(toLength: 24, withPad: " ", startingAt: 0))" +
                  "\(String(format: "%6.2f", m))ms  (\(String(format: "%4.1f", pct))%)  \(bar)")
        }

        print()
        print("  Process (excl. wait)   \(String(format: "%6.2f", totalMed))ms " +
              "→ \(String(format: "%.1f", 1000/totalMed)) FPS")
        print("  p95:                   \(String(format: "%.2f", pct95(t_total) * 1000))ms")

        let waitMed = median(t_wait) * 1000
        print("\n  Frame wait median:     \(String(format: "%.2f", waitMed))ms")

        print("\nComparison:")
        print("  Python CPU baseline:    503.6ms  ( 2.0 FPS)")
        print("  Python optimized:        52.8ms  (18.9 FPS)")
        print("  Swift sync (prev):       46.7ms  (21.4 FPS)")
        print("  Swift async:         \(String(format: "%8.1f", totalMed))ms  " +
              "(\(String(format: "%.1f", 1000/totalMed)) FPS)")
}

// Top-level entry point
let semaphore = DispatchSemaphore(value: 0)
Task {
    do {
        try await runPipeline()
    } catch {
        print("Error: \(error)")
    }
    semaphore.signal()
}
semaphore.wait()
