import Foundation
import AVFoundation
import Vision
import CoreVideo
import SwiftUI

/// Thread-safe config snapshot read from videoQueue
final class PipelineConfig: @unchecked Sendable {
    private let lock = NSLock()
    private var _enableSeg = true
    private var _enableBody = true
    private var _enableHands = true
    private var _enableFace = true
    private var _segQuality: VNGeneratePersonSegmentationRequest.QualityLevel = .balanced

    var enableSeg: Bool { lock.withLock { _enableSeg } }
    var enableBody: Bool { lock.withLock { _enableBody } }
    var enableHands: Bool { lock.withLock { _enableHands } }
    var enableFace: Bool { lock.withLock { _enableFace } }
    var segQuality: VNGeneratePersonSegmentationRequest.QualityLevel { lock.withLock { _segQuality } }

    func update(seg: Bool, body: Bool, hands: Bool, face: Bool,
                quality: VNGeneratePersonSegmentationRequest.QualityLevel) {
        lock.withLock {
            _enableSeg = seg; _enableBody = body
            _enableHands = hands; _enableFace = face
            _segQuality = quality
        }
    }
}

@MainActor
class VisionPipelineEngine: NSObject, ObservableObject {
    @Published var latestResult: VisionFrameResult?
    @Published var fps: Double = 0
    @Published var isRunning = false
    @Published var imageSize: CGSize = CGSize(width: 1920, height: 1080)

    // Per-request timing (ms)
    @Published var segMs: Double = 0
    @Published var bodyMs: Double = 0
    @Published var handMs: Double = 0
    @Published var faceMs: Double = 0

    // UI-bound toggles
    @Published var enableSeg = true { didSet { syncConfig() } }
    @Published var enableBody = true { didSet { syncConfig() } }
    @Published var enableHands = true { didSet { syncConfig() } }
    @Published var enableFace = true { didSet { syncConfig() } }
    @Published var segQuality: VNGeneratePersonSegmentationRequest.QualityLevel = .balanced { didSet { syncConfig() } }

    // Thread-safe config for videoQueue
    let config = PipelineConfig()

    private func syncConfig() {
        config.update(seg: enableSeg, body: enableBody,
                      hands: enableHands, face: enableFace,
                      quality: segQuality)
    }

    private var captureSession: AVCaptureSession?
    private let videoQueue = DispatchQueue(label: "vision-video-capture")

    // Vision requests — created once, reused across frames (accessed from videoQueue)
    private let segRequest: VNGeneratePersonSegmentationRequest
    private let bodyPoseRequest: VNDetectHumanBodyPoseRequest
    private let handPoseRequest: VNDetectHumanHandPoseRequest
    private let faceLandmarksRequest: VNDetectFaceLandmarksRequest

    // Timing
    private var frameTimes: [Double] = []
    private let maxFrameTimes = 60

    override init() {
        segRequest = VNGeneratePersonSegmentationRequest()
        segRequest.qualityLevel = .balanced
        segRequest.outputPixelFormat = kCVPixelFormatType_OneComponent8

        bodyPoseRequest = VNDetectHumanBodyPoseRequest()

        handPoseRequest = VNDetectHumanHandPoseRequest()
        handPoseRequest.maximumHandCount = 2

        faceLandmarksRequest = VNDetectFaceLandmarksRequest()
        faceLandmarksRequest.constellation = .constellation76Points

        super.init()
    }

    func start() throws {
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
        isRunning = true
    }

    func stop() {
        captureSession?.stopRunning()
        isRunning = false
    }

    // MARK: - Process Frame (called on videoQueue)

    nonisolated private func processFrame(_ pixelBuffer: CVPixelBuffer) {
        let t0 = CFAbsoluteTimeGetCurrent()
        let imgW = CVPixelBufferGetWidth(pixelBuffer)
        let imgH = CVPixelBufferGetHeight(pixelBuffer)

        // Snapshot config
        let doSeg = config.enableSeg
        let doBody = config.enableBody
        let doHands = config.enableHands
        let doFace = config.enableFace
        let quality = config.segQuality

        if segRequest.qualityLevel != quality {
            segRequest.qualityLevel = quality
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])

        var tSeg: Double = 0, tBody: Double = 0, tHand: Double = 0, tFace: Double = 0

        var maskPB: CVPixelBuffer? = nil
        if doSeg {
            let s = CFAbsoluteTimeGetCurrent()
            try? handler.perform([segRequest])
            tSeg = (CFAbsoluteTimeGetCurrent() - s) * 1000
            maskPB = (segRequest.results?.first as? VNPixelBufferObservation)?.pixelBuffer
        }

        var bodies: [BodyPoseData] = []
        if doBody {
            let s = CFAbsoluteTimeGetCurrent()
            try? handler.perform([bodyPoseRequest])
            tBody = (CFAbsoluteTimeGetCurrent() - s) * 1000
            bodies = extractBodyPoses(imgW: imgW, imgH: imgH)
        }

        var hands: [HandPoseData] = []
        if doHands {
            let s = CFAbsoluteTimeGetCurrent()
            try? handler.perform([handPoseRequest])
            tHand = (CFAbsoluteTimeGetCurrent() - s) * 1000
            hands = extractHandPoses(imgW: imgW, imgH: imgH)
        }

        var faces: [FaceLandmarkData] = []
        if doFace {
            let s = CFAbsoluteTimeGetCurrent()
            try? handler.perform([faceLandmarksRequest])
            tFace = (CFAbsoluteTimeGetCurrent() - s) * 1000
            faces = extractFaceLandmarks(imgW: imgW, imgH: imgH)
        }

        let tEnd = CFAbsoluteTimeGetCurrent()
        let frameMs = (tEnd - t0) * 1000

        let result = VisionFrameResult(
            maskPixelBuffer: maskPB,
            bodyPoses: bodies,
            handPoses: hands,
            faceLandmarks: faces,
            frameTimeMs: frameMs
        )

        Task { @MainActor in
            self.imageSize = CGSize(width: imgW, height: imgH)
            self.latestResult = result
            self.segMs = tSeg
            self.bodyMs = tBody
            self.handMs = tHand
            self.faceMs = tFace
            self.frameTimes.append(frameMs)
            if self.frameTimes.count > self.maxFrameTimes { self.frameTimes.removeFirst() }
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
