import Foundation
import AVFoundation
import Vision
import CoreVideo
import SwiftUI
import Metal
import Syphon

/// Thread-safe config snapshot read from videoQueue
final class PipelineConfig: @unchecked Sendable {
    private let lock = NSLock()
    private var _enableSeg = true
    private var _enableBody = false
    private var _enableHands = false
    private var _enableFace = false
    private var _segQuality: VNGeneratePersonSegmentationRequest.QualityLevel = .balanced
    private var _batchMode = false
    private var _enableSyphon = false
    private var _maskThreshold: UInt8 = 128
    private var _enableContours = false

    var enableSeg: Bool { lock.withLock { _enableSeg } }
    var enableBody: Bool { lock.withLock { _enableBody } }
    var enableHands: Bool { lock.withLock { _enableHands } }
    var enableFace: Bool { lock.withLock { _enableFace } }
    var segQuality: VNGeneratePersonSegmentationRequest.QualityLevel { lock.withLock { _segQuality } }
    var batchMode: Bool { lock.withLock { _batchMode } }
    var enableSyphon: Bool { lock.withLock { _enableSyphon } }
    var maskThreshold: UInt8 { lock.withLock { _maskThreshold } }
    var enableContours: Bool { lock.withLock { _enableContours } }

    func update(seg: Bool, body: Bool, hands: Bool, face: Bool,
                quality: VNGeneratePersonSegmentationRequest.QualityLevel,
                batch: Bool, syphon: Bool, threshold: UInt8, contours: Bool) {
        lock.withLock {
            _enableSeg = seg; _enableBody = body
            _enableHands = hands; _enableFace = face
            _segQuality = quality; _batchMode = batch
            _enableSyphon = syphon; _maskThreshold = threshold
            _enableContours = contours
        }
    }
}

struct CameraInfo: Identifiable, Hashable {
    let id: String
    let name: String
}

@MainActor
class VisionPipelineEngine: NSObject, ObservableObject {
    @Published var latestResult: VisionFrameResult?
    @Published var fps: Double = 0
    @Published var isRunning = false
    @Published var imageSize: CGSize = CGSize(width: 1920, height: 1080)

    // Per-request timing — rolling averages (ms)
    @Published var segMs: Double = 0
    @Published var bodyMs: Double = 0
    @Published var handMs: Double = 0
    @Published var faceMs: Double = 0
    @Published var contourMs: Double = 0
    private var segHistory: [Double] = []
    private var bodyHistory: [Double] = []
    private var handHistory: [Double] = []
    private var faceHistory: [Double] = []
    private var contourHistory: [Double] = []
    private let avgWindow = 10

    // UI-bound toggles
    @Published var enableSeg = true
    @Published var enableBody = false
    @Published var enableHands = false
    @Published var enableFace = false
    @Published var enableContours = false
    @Published var segQualityIndex = 1  // 0=fast, 1=balanced, 2=accurate
    @Published var batchMode = false

    // Syphon + masking
    @Published var enableSyphon = false
    @Published var maskThreshold: Float = 0.5  // 0-1, mapped to 0-255

    // Camera selection
    @Published var availableCameras: [CameraInfo] = []
    @Published var selectedCameraId: String = ""

    // Thread-safe config for videoQueue
    let config = PipelineConfig()

    func syncConfig() {
        let q: VNGeneratePersonSegmentationRequest.QualityLevel
        switch segQualityIndex {
        case 0: q = .fast
        case 2: q = .accurate
        default: q = .balanced
        }
        config.update(seg: enableSeg, body: enableBody,
                      hands: enableHands, face: enableFace,
                      quality: q, batch: batchMode,
                      syphon: enableSyphon,
                      threshold: UInt8(min(255, max(0, maskThreshold * 255))),
                      contours: enableContours)
    }

    private var captureSession: AVCaptureSession?
    private var currentInput: AVCaptureDeviceInput?
    private let videoQueue = DispatchQueue(label: "vision-video-capture", qos: .userInitiated)

    // Vision requests
    private let segRequest: VNGeneratePersonSegmentationRequest
    private let bodyPoseRequest: VNDetectHumanBodyPoseRequest
    private let handPoseRequest: VNDetectHumanHandPoseRequest
    private let faceLandmarksRequest: VNDetectFaceLandmarksRequest
    private let contourRequest: VNDetectContoursRequest

    // Sequence handlers
    private let segSeqHandler = VNSequenceRequestHandler()
    private let bodySeqHandler = VNSequenceRequestHandler()
    private let faceSeqHandler = VNSequenceRequestHandler()

    // Timing
    private var frameTimes: [Double] = []

    // Metal + Syphon (accessed from videoQueue)
    nonisolated(unsafe) private let metalDevice: MTLDevice
    nonisolated(unsafe) private let commandQueue: MTLCommandQueue
    nonisolated(unsafe) private var textureCache: CVMetalTextureCache?
    nonisolated(unsafe) private var syphonServer: SyphonMetalServer?
    nonisolated(unsafe) private var maskedBuffer: CVPixelBuffer?

    override init() {
        segRequest = VNGeneratePersonSegmentationRequest()
        segRequest.qualityLevel = .balanced
        segRequest.outputPixelFormat = kCVPixelFormatType_OneComponent8

        bodyPoseRequest = VNDetectHumanBodyPoseRequest()

        handPoseRequest = VNDetectHumanHandPoseRequest()
        handPoseRequest.maximumHandCount = 2

        faceLandmarksRequest = VNDetectFaceLandmarksRequest()
        faceLandmarksRequest.constellation = .constellation76Points

        contourRequest = VNDetectContoursRequest()
        contourRequest.contrastAdjustment = 2.0

        metalDevice = MTLCreateSystemDefaultDevice()!
        commandQueue = metalDevice.makeCommandQueue()!

        super.init()

        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, metalDevice, nil, &textureCache)
        refreshCameraList()
    }

    // MARK: - Camera Management

    func refreshCameraList() {
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .external],
            mediaType: .video, position: .unspecified)
        availableCameras = discovery.devices.map { CameraInfo(id: $0.uniqueID, name: $0.localizedName) }
        if selectedCameraId.isEmpty, let first = availableCameras.first {
            selectedCameraId = first.id
        }
    }

    func start() throws {
        let session = AVCaptureSession()
        session.sessionPreset = .hd1920x1080

        guard let device = cameraDevice(for: selectedCameraId),
              let input = try? AVCaptureDeviceInput(device: device)
        else { throw NSError(domain: "Camera", code: 1, userInfo: [NSLocalizedDescriptionKey: "No camera"]) }

        session.addInput(input)
        currentInput = input

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: videoQueue)
        session.addOutput(output)

        captureSession = session
        session.startRunning()
        isRunning = true
    }

    func switchCamera(to id: String) {
        guard let session = captureSession,
              let device = cameraDevice(for: id),
              let newInput = try? AVCaptureDeviceInput(device: device)
        else { return }

        session.beginConfiguration()
        if let old = currentInput { session.removeInput(old) }
        if session.canAddInput(newInput) {
            session.addInput(newInput)
            currentInput = newInput
            selectedCameraId = id
        }
        session.commitConfiguration()
    }

    private func cameraDevice(for id: String) -> AVCaptureDevice? {
        AVCaptureDevice(uniqueID: id) ?? AVCaptureDevice.default(for: .video)
    }

    func stop() {
        captureSession?.stopRunning()
        syphonServer?.stop()
        isRunning = false
    }

    // MARK: - Syphon

    nonisolated private func ensureSyphonServer() {
        if syphonServer == nil {
            syphonServer = SyphonMetalServer(name: "VisionApp", device: metalDevice, options: nil)
        }
    }

    nonisolated private func publishMaskedFrame(camera: CVPixelBuffer, mask: CVPixelBuffer) {
        let threshold = config.maskThreshold
        let imgW = CVPixelBufferGetWidth(camera)
        let imgH = CVPixelBufferGetHeight(camera)
        let maskW = CVPixelBufferGetWidth(mask)
        let maskH = CVPixelBufferGetHeight(mask)

        // Create or reuse output buffer
        if maskedBuffer == nil
            || CVPixelBufferGetWidth(maskedBuffer!) != imgW
            || CVPixelBufferGetHeight(maskedBuffer!) != imgH {
            var pb: CVPixelBuffer?
            CVPixelBufferCreate(kCFAllocatorDefault, imgW, imgH,
                                kCVPixelFormatType_32BGRA,
                                [kCVPixelBufferIOSurfacePropertiesKey: [:] as [String: Any],
                                 kCVPixelBufferMetalCompatibilityKey: true] as CFDictionary,
                                &pb)
            maskedBuffer = pb
        }
        guard let outPB = maskedBuffer else { return }

        CVPixelBufferLockBaseAddress(camera, .readOnly)
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        CVPixelBufferLockBaseAddress(outPB, [])

        let camBase = CVPixelBufferGetBaseAddress(camera)!.assumingMemoryBound(to: UInt8.self)
        let camRowBytes = CVPixelBufferGetBytesPerRow(camera)
        let maskBase = CVPixelBufferGetBaseAddress(mask)!.assumingMemoryBound(to: UInt8.self)
        let maskRowBytes = CVPixelBufferGetBytesPerRow(mask)
        let outBase = CVPixelBufferGetBaseAddress(outPB)!.assumingMemoryBound(to: UInt8.self)
        let outRowBytes = CVPixelBufferGetBytesPerRow(outPB)

        // Apply mask with nearest-neighbor upscale
        for y in 0..<imgH {
            let my = y * maskH / imgH
            let outRow = outBase.advanced(by: y * outRowBytes)
            let camRow = camBase.advanced(by: y * camRowBytes)
            for x in 0..<imgW {
                let mx = x * maskW / imgW
                let maskVal = maskBase[my * maskRowBytes + mx]
                let i = x * 4
                if maskVal > threshold {
                    outRow[i] = camRow[i]
                    outRow[i+1] = camRow[i+1]
                    outRow[i+2] = camRow[i+2]
                    outRow[i+3] = 255
                } else {
                    outRow[i] = 0; outRow[i+1] = 0; outRow[i+2] = 0; outRow[i+3] = 0
                }
            }
        }

        CVPixelBufferUnlockBaseAddress(camera, .readOnly)
        CVPixelBufferUnlockBaseAddress(mask, .readOnly)
        CVPixelBufferUnlockBaseAddress(outPB, [])

        // Publish via Syphon
        guard let cache = textureCache else { return }
        var cvTex: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, outPB, nil,
            .bgra8Unorm, imgW, imgH, 0, &cvTex)
        guard status == kCVReturnSuccess, let cvTex = cvTex,
              let texture = CVMetalTextureGetTexture(cvTex),
              let cmdBuf = commandQueue.makeCommandBuffer()
        else { return }

        syphonServer?.publishFrameTexture(texture, on: cmdBuf,
                                           imageRegion: NSRect(x: 0, y: 0, width: imgW, height: imgH),
                                           flipped: true)
        cmdBuf.commit()
    }

    // MARK: - Process Frame (called on videoQueue)

    nonisolated private func processFrame(_ pixelBuffer: CVPixelBuffer) {
        let t0 = CFAbsoluteTimeGetCurrent()
        let imgW = CVPixelBufferGetWidth(pixelBuffer)
        let imgH = CVPixelBufferGetHeight(pixelBuffer)

        let doSeg = config.enableSeg
        let doBody = config.enableBody
        let doHands = config.enableHands
        let doFace = config.enableFace
        let doContours = config.enableContours
        let quality = config.segQuality
        let batch = config.batchMode
        let doSyphon = config.enableSyphon

        if segRequest.qualityLevel != quality {
            segRequest.qualityLevel = quality
        }

        var tSeg: Double = 0, tBody: Double = 0, tHand: Double = 0, tFace: Double = 0, tContour: Double = 0
        var maskPB: CVPixelBuffer? = nil
        var contourPath: CGPath? = nil
        var bodies: [BodyPoseData] = []
        var hands: [HandPoseData] = []
        var faces: [FaceLandmarkData] = []

        if batch {
            var requests: [VNRequest] = []
            if doSeg || doSyphon || doContours { requests.append(segRequest) }
            if doBody { requests.append(bodyPoseRequest) }
            if doHands { requests.append(handPoseRequest) }
            if doFace { requests.append(faceLandmarksRequest) }

            if !requests.isEmpty {
                let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
                try? handler.perform(requests)
            }

            if doSeg || doSyphon || doContours {
                maskPB = (segRequest.results?.first as? VNPixelBufferObservation)?.pixelBuffer
            }
            if doBody { bodies = extractBodyPoses(imgW: imgW, imgH: imgH) }
            if doHands { hands = extractHandPoses(imgW: imgW, imgH: imgH) }
            if doFace { faces = extractFaceLandmarks(imgW: imgW, imgH: imgH) }
        } else {
            // Segmentation needed for viz, syphon, or contours
            if doSeg || doSyphon || doContours {
                let s = CFAbsoluteTimeGetCurrent()
                try? segSeqHandler.perform([segRequest], on: pixelBuffer)
                tSeg = (CFAbsoluteTimeGetCurrent() - s) * 1000
                maskPB = (segRequest.results?.first as? VNPixelBufferObservation)?.pixelBuffer
            }

            if doBody {
                let s = CFAbsoluteTimeGetCurrent()
                try? bodySeqHandler.perform([bodyPoseRequest], on: pixelBuffer)
                tBody = (CFAbsoluteTimeGetCurrent() - s) * 1000
                bodies = extractBodyPoses(imgW: imgW, imgH: imgH)
            }

            if doHands {
                let s = CFAbsoluteTimeGetCurrent()
                let handHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
                try? handHandler.perform([handPoseRequest])
                tHand = (CFAbsoluteTimeGetCurrent() - s) * 1000
                hands = extractHandPoses(imgW: imgW, imgH: imgH)
            }

            if doFace {
                let s = CFAbsoluteTimeGetCurrent()
                try? faceSeqHandler.perform([faceLandmarksRequest], on: pixelBuffer)
                tFace = (CFAbsoluteTimeGetCurrent() - s) * 1000
                faces = extractFaceLandmarks(imgW: imgW, imgH: imgH)
            }
        }

        // Contour detection: blit mask onto larger canvas with 20px zero border,
        // then remap contour coords back to original mask space
        if doContours, let mask = maskPB {
            let s = CFAbsoluteTimeGetCurrent()
            let mW = CVPixelBufferGetWidth(mask)
            let mH = CVPixelBufferGetHeight(mask)
            let border = 20
            let padW = mW + 2 * border
            let padH = mH + 2 * border

            var paddedBuf: CVPixelBuffer?
            CVPixelBufferCreate(kCFAllocatorDefault, padW, padH,
                                kCVPixelFormatType_OneComponent8, nil, &paddedBuf)
            if let pb = paddedBuf {
                CVPixelBufferLockBaseAddress(mask, .readOnly)
                CVPixelBufferLockBaseAddress(pb, [])
                let src = CVPixelBufferGetBaseAddress(mask)!.assumingMemoryBound(to: UInt8.self)
                let srcStride = CVPixelBufferGetBytesPerRow(mask)
                let dst = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
                let dstStride = CVPixelBufferGetBytesPerRow(pb)
                // Zero fill, then blit mask at offset (border, border)
                memset(dst, 0, dstStride * padH)
                for y in 0..<mH {
                    memcpy(dst.advanced(by: (y + border) * dstStride + border),
                           src.advanced(by: y * srcStride),
                           mW)
                }
                CVPixelBufferUnlockBaseAddress(mask, .readOnly)
                CVPixelBufferUnlockBaseAddress(pb, [])

                let contourHandler = VNImageRequestHandler(cvPixelBuffer: pb, options: [:])
                try? contourHandler.perform([contourRequest])
                tContour = (CFAbsoluteTimeGetCurrent() - s) * 1000

                if let obs = contourRequest.results?.first as? VNContoursObservation {
                    // Remap from padded normalized coords to original mask normalized coords
                    var remap = CGAffineTransform(
                        a: CGFloat(padW) / CGFloat(mW), b: 0,
                        c: 0, d: CGFloat(padH) / CGFloat(mH),
                        tx: -CGFloat(border) / CGFloat(mW),
                        ty: -CGFloat(border) / CGFloat(mH))
                    contourPath = obs.normalizedPath.copy(using: &remap)
                }
            }
        }

        // Syphon output
        if doSyphon, let mask = maskPB {
            ensureSyphonServer()
            publishMaskedFrame(camera: pixelBuffer, mask: mask)
        } else if !doSyphon, syphonServer != nil {
            syphonServer?.stop()
            syphonServer = nil
        }

        let tEnd = CFAbsoluteTimeGetCurrent()
        let frameMs = (tEnd - t0) * 1000

        let result = VisionFrameResult(
            maskPixelBuffer: doSeg ? maskPB : nil,
            contourPath: doContours ? contourPath : nil,
            bodyPoses: bodies,
            handPoses: hands,
            faceLandmarks: faces,
            frameTimeMs: frameMs
        )

        Task { @MainActor in
            self.imageSize = CGSize(width: imgW, height: imgH)
            self.latestResult = result

            self.segHistory.append(tSeg)
            self.bodyHistory.append(tBody)
            self.handHistory.append(tHand)
            self.faceHistory.append(tFace)
            self.contourHistory.append(tContour)
            if self.segHistory.count > self.avgWindow { self.segHistory.removeFirst() }
            if self.bodyHistory.count > self.avgWindow { self.bodyHistory.removeFirst() }
            if self.handHistory.count > self.avgWindow { self.handHistory.removeFirst() }
            if self.faceHistory.count > self.avgWindow { self.faceHistory.removeFirst() }
            if self.contourHistory.count > self.avgWindow { self.contourHistory.removeFirst() }
            self.segMs = self.segHistory.reduce(0, +) / Double(self.segHistory.count)
            self.bodyMs = self.bodyHistory.reduce(0, +) / Double(self.bodyHistory.count)
            self.handMs = self.handHistory.reduce(0, +) / Double(self.handHistory.count)
            self.faceMs = self.faceHistory.reduce(0, +) / Double(self.faceHistory.count)
            self.contourMs = self.contourHistory.reduce(0, +) / Double(self.contourHistory.count)

            self.frameTimes.append(frameMs)
            if self.frameTimes.count > self.avgWindow { self.frameTimes.removeFirst() }
            self.fps = 1000.0 / (self.frameTimes.reduce(0, +) / Double(self.frameTimes.count))
        }
    }

    // MARK: - Result Extraction

    nonisolated private func extractBodyPoses(imgW: Int, imgH: Int) -> [BodyPoseData] {
        guard let observations = bodyPoseRequest.results as? [VNHumanBodyPoseObservation] else { return [] }
        return observations.compactMap { obs in
            guard let allPoints = try? obs.recognizedPoints(.all) else { return nil }
            var joints: [String: (CGPoint, Float)] = [:]
            for (key, point) in allPoints {
                let px = point.location.x * CGFloat(imgW)
                let py = (1.0 - point.location.y) * CGFloat(imgH)
                joints[key.rawValue.rawValue] = (CGPoint(x: px, y: py), Float(point.confidence))
            }
            return BodyPoseData(joints: joints)
        }
    }

    nonisolated private func extractHandPoses(imgW: Int, imgH: Int) -> [HandPoseData] {
        guard let observations = handPoseRequest.results as? [VNHumanHandPoseObservation] else { return [] }
        return observations.compactMap { obs in
            guard let allPoints = try? obs.recognizedPoints(.all) else { return nil }
            var joints: [String: (CGPoint, Float)] = [:]
            for (key, point) in allPoints {
                let px = point.location.x * CGFloat(imgW)
                let py = (1.0 - point.location.y) * CGFloat(imgH)
                joints[key.rawValue.rawValue] = (CGPoint(x: px, y: py), Float(point.confidence))
            }
            let chirality: HandChirality
            switch obs.chirality {
            case .left: chirality = .left
            case .right: chirality = .right
            default: chirality = .unknown
            }
            return HandPoseData(joints: joints, chirality: chirality)
        }
    }

    nonisolated private func extractFaceLandmarks(imgW: Int, imgH: Int) -> [FaceLandmarkData] {
        guard let observations = faceLandmarksRequest.results as? [VNFaceObservation] else { return [] }
        return observations.compactMap { obs in
            guard let landmarks = obs.landmarks,
                  let allPoints = landmarks.allPoints else { return nil }

            let faceBBox = obs.boundingBox
            let points: [CGPoint] = (0..<allPoints.pointCount).map { i in
                let pt = allPoints.normalizedPoints[i]
                let imgNormX = faceBBox.origin.x + CGFloat(pt.x) * faceBBox.width
                let imgNormY = faceBBox.origin.y + CGFloat(pt.y) * faceBBox.height
                let px = imgNormX * CGFloat(imgW)
                let py = (1.0 - imgNormY) * CGFloat(imgH)
                return CGPoint(x: px, y: py)
            }

            let bboxPixel = CGRect(
                x: faceBBox.origin.x * CGFloat(imgW),
                y: (1.0 - faceBBox.origin.y - faceBBox.height) * CGFloat(imgH),
                width: faceBBox.width * CGFloat(imgW),
                height: faceBBox.height * CGFloat(imgH)
            )

            return FaceLandmarkData(allPoints: points, boundingBox: bboxPixel)
        }
    }
}

// MARK: - Camera Delegate

extension VisionPipelineEngine: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput,
                                    didOutput sampleBuffer: CMSampleBuffer,
                                    from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        processFrame(pb)
    }
}
