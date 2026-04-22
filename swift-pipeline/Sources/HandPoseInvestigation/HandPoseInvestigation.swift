import AVFoundation
import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import ImageIO
import Vision

public enum RequestLifetime: String, Codable, CaseIterable {
    case fresh
    case reused
}

public enum HandlerKind: String, Codable, CaseIterable {
    case image
    case sequence
}

public enum RequestSetKind: String, Codable, CaseIterable {
    case handOnly = "hand_only"
    case segThenHand = "seg_then_hand"
    case visionBatch = "vision_batch"
}

public enum VisionInputOrientation: String, Codable, CaseIterable {
    case unspecified
    case up
    case down
    case left
    case right

    var cgOrientation: CGImagePropertyOrientation? {
        switch self {
        case .unspecified: return nil
        case .up: return .up
        case .down: return .down
        case .left: return .left
        case .right: return .right
        }
    }
}

public struct HandPoseRunConfig: Codable {
    public var runId: String
    public var name: String
    public var requestLifetime: RequestLifetime
    public var handler: HandlerKind
    public var requestSet: RequestSetKind
    public var usesCPUOnly: Bool
    public var maximumHandCount: Int
    public var revision: Int?
    public var inputOrientation: VisionInputOrientation
    public var startSeconds: Double
    public var durationSeconds: Double?
    public var maxFrames: Int?
    public var frameStride: Int

    public init(
        runId: String,
        name: String,
        requestLifetime: RequestLifetime,
        handler: HandlerKind,
        requestSet: RequestSetKind,
        usesCPUOnly: Bool,
        maximumHandCount: Int,
        revision: Int?,
        inputOrientation: VisionInputOrientation,
        startSeconds: Double,
        durationSeconds: Double?,
        maxFrames: Int?,
        frameStride: Int
    ) {
        self.runId = runId
        self.name = name
        self.requestLifetime = requestLifetime
        self.handler = handler
        self.requestSet = requestSet
        self.usesCPUOnly = usesCPUOnly
        self.maximumHandCount = maximumHandCount
        self.revision = revision
        self.inputOrientation = inputOrientation
        self.startSeconds = startSeconds
        self.durationSeconds = durationSeconds
        self.maxFrames = maxFrames
        self.frameStride = frameStride
    }
}

public struct VisionErrorRecord: Codable {
    public let domain: String
    public let code: Int
    public let description: String
}

public struct HandJointRecord: Codable {
    public let name: String
    public let confidence: Float
    public let confidenceClass: String
    public let normalizedX: Double
    public let normalizedY: Double
    public let pixelX: Double
    public let pixelY: Double
    public let coordinateClass: String
}

public struct ObservationSummary: Codable {
    public let jointCount: Int
    public let normalConfidenceCount: Int
    public let zeroConfidenceCount: Int
    public let outOfRangeHighConfidenceCount: Int
    public let negativeConfidenceCount: Int
    public let nonFiniteConfidenceCount: Int
    public let invalidCoordinateCount: Int
    public let minConfidence: Float?
    public let maxConfidence: Float?
}

public struct HandObservationRecord: Codable {
    public let index: Int
    public let chirality: String
    public let extractError: String?
    public let summary: ObservationSummary
    public let joints: [HandJointRecord]
}

public struct FrameSummary: Codable {
    public let observationCount: Int
    public let jointCount: Int
    public let normalConfidenceCount: Int
    public let zeroConfidenceCount: Int
    public let outOfRangeHighConfidenceCount: Int
    public let negativeConfidenceCount: Int
    public let nonFiniteConfidenceCount: Int
    public let invalidCoordinateCount: Int
}

public struct HandPoseFrameRecord: Codable {
    public let schemaVersion: Int
    public let recordType: String
    public let runId: String
    public let config: HandPoseRunConfig
    public let videoPath: String
    public let frameIndex: Int
    public let sourceFrameIndex: Int
    public let timestampSeconds: Double
    public let imageWidth: Int
    public let imageHeight: Int
    public let effectiveRevision: Int
    public let durationMs: Double
    public let error: VisionErrorRecord?
    public let observations: [HandObservationRecord]
    public let summary: FrameSummary
}

public struct HandPoseRunSummary: Codable {
    public let schemaVersion: Int
    public let recordType: String
    public let runId: String
    public let config: HandPoseRunConfig
    public let videoPath: String
    public let effectiveRevision: Int
    public let framesProcessed: Int
    public let framesWithError: Int
    public let framesWithObservations: Int
    public let totalObservations: Int
    public let totalJoints: Int
    public let totalOutOfRangeHighConfidence: Int
    public let totalInvalidCoordinates: Int
    public let elapsedMs: Double
}

public final class JSONLineWriter {
    private let handle: FileHandle
    private let shouldClose: Bool
    private let encoder: JSONEncoder

    public init(path: String?) throws {
        encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        encoder.nonConformingFloatEncodingStrategy = .convertToString(
            positiveInfinity: "Infinity",
            negativeInfinity: "-Infinity",
            nan: "NaN"
        )

        guard let path, path != "-" else {
            handle = .standardOutput
            shouldClose = false
            return
        }

        let url = URL(fileURLWithPath: path)
        let parent = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try FileHandle(forWritingTo: url)
        shouldClose = true
    }

    deinit {
        if shouldClose {
            try? handle.close()
        }
    }

    public func write<T: Encodable>(_ value: T) throws {
        var data = try encoder.encode(value)
        data.append(0x0A)
        try handle.write(contentsOf: data)
    }
}

public final class JSONFileWriter {
    private let encoder: JSONEncoder

    public init() {
        encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.nonConformingFloatEncodingStrategy = .convertToString(
            positiveInfinity: "Infinity",
            negativeInfinity: "-Infinity",
            nan: "NaN"
        )
    }

    public func write<T: Encodable>(_ value: T, to path: String) throws {
        let url = URL(fileURLWithPath: path)
        let parent = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let data = try encoder.encode(value)
        try data.write(to: url, options: .atomic)
    }
}

public struct VideoFrame {
    public let sourceIndex: Int
    public let processedIndex: Int
    public let timestampSeconds: Double
    public let pixelBuffer: CVPixelBuffer
}

public final class VideoFrameReader {
    public static func read(
        videoURL: URL,
        startSeconds: Double,
        durationSeconds: Double?,
        maxFrames: Int?,
        frameStride: Int,
        body: (VideoFrame) throws -> Void
    ) throws {
        let asset = AVURLAsset(url: videoURL)
        guard let track = asset.tracks(withMediaType: .video).first else {
            throw InvestigationError("No video track found in \(videoURL.path)")
        }

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
        )
        output.alwaysCopiesSampleData = false

        guard reader.canAdd(output) else {
            throw InvestigationError("Cannot add AVAssetReaderTrackOutput for \(videoURL.path)")
        }
        reader.add(output)
        if startSeconds > 0 || durationSeconds != nil {
            let start = CMTime(seconds: startSeconds, preferredTimescale: 600)
            let duration: CMTime
            if let durationSeconds {
                duration = CMTime(seconds: durationSeconds, preferredTimescale: 600)
            } else {
                duration = asset.duration.isNumeric && asset.duration > start
                    ? CMTimeSubtract(asset.duration, start)
                    : .positiveInfinity
            }
            reader.timeRange = CMTimeRange(start: start, duration: duration)
        }

        guard reader.startReading() else {
            throw reader.error ?? InvestigationError("AVAssetReader failed to start")
        }

        let stride = max(1, frameStride)
        var sourceIndex = 0
        var processedIndex = 0

        while reader.status == .reading {
            guard let sampleBuffer = output.copyNextSampleBuffer() else { break }
            defer { sourceIndex += 1 }

            if sourceIndex % stride != 0 { continue }
            if let maxFrames, processedIndex >= maxFrames { break }
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { continue }

            let time = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            let timestamp = time.isValid ? CMTimeGetSeconds(time) : Double(sourceIndex)
            let frame = VideoFrame(
                sourceIndex: sourceIndex,
                processedIndex: processedIndex,
                timestampSeconds: timestamp,
                pixelBuffer: pixelBuffer
            )
            try body(frame)
            processedIndex += 1
        }

        if reader.status == .failed {
            throw reader.error ?? InvestigationError("AVAssetReader failed while reading")
        }
        reader.cancelReading()
    }
}

public final class HandPoseReplayRunner {
    private let config: HandPoseRunConfig
    private let videoPath: String
    private let effectiveRevision: Int
    private var reusedHandRequest: VNDetectHumanHandPoseRequest?
    private let sequenceHandler = VNSequenceRequestHandler()
    private let preflightSequenceHandler = VNSequenceRequestHandler()
    private let segmentationRequest: VNGeneratePersonSegmentationRequest
    private let bodyRequest: VNDetectHumanBodyPoseRequest
    private let faceRequest: VNDetectFaceLandmarksRequest

    public init(config: HandPoseRunConfig, videoPath: String) throws {
        self.config = config
        self.videoPath = videoPath

        if let revision = config.revision,
           !VNDetectHumanHandPoseRequest.supportedRevisions.contains(revision) {
            throw InvestigationError(
                "Unsupported VNDetectHumanHandPoseRequest revision \(revision). " +
                    "Supported revisions: \(Array(VNDetectHumanHandPoseRequest.supportedRevisions))"
            )
        }

        let probe = VNDetectHumanHandPoseRequest()
        effectiveRevision = config.revision ?? probe.revision

        segmentationRequest = VNGeneratePersonSegmentationRequest()
        segmentationRequest.qualityLevel = .balanced
        segmentationRequest.outputPixelFormat = kCVPixelFormatType_OneComponent8

        bodyRequest = VNDetectHumanBodyPoseRequest()
        faceRequest = VNDetectFaceLandmarksRequest()
        faceRequest.constellation = .constellation76Points
    }

    public func run(videoURL: URL, writer: JSONLineWriter) throws -> HandPoseRunSummary {
        let start = DispatchTime.now().uptimeNanoseconds
        var framesProcessed = 0
        var framesWithError = 0
        var framesWithObservations = 0
        var totalObservations = 0
        var totalJoints = 0
        var totalOutOfRangeHighConfidence = 0
        var totalInvalidCoordinates = 0

        try VideoFrameReader.read(
            videoURL: videoURL,
            startSeconds: config.startSeconds,
            durationSeconds: config.durationSeconds,
            maxFrames: config.maxFrames,
            frameStride: config.frameStride
        ) { frame in
            let record = try self.process(
                pixelBuffer: frame.pixelBuffer,
                sourceFrameIndex: frame.sourceIndex,
                processedIndex: frame.processedIndex,
                timestampSeconds: frame.timestampSeconds
            )
            try writer.write(record)

            framesProcessed += 1
            if record.error != nil { framesWithError += 1 }
            if !record.observations.isEmpty { framesWithObservations += 1 }
            totalObservations += record.summary.observationCount
            totalJoints += record.summary.jointCount
            totalOutOfRangeHighConfidence += record.summary.outOfRangeHighConfidenceCount
            totalInvalidCoordinates += record.summary.invalidCoordinateCount
        }

        let elapsedMs = elapsedMilliseconds(since: start)
        return HandPoseRunSummary(
            schemaVersion: 1,
            recordType: "summary",
            runId: config.runId,
            config: config,
            videoPath: videoPath,
            effectiveRevision: effectiveRevision,
            framesProcessed: framesProcessed,
            framesWithError: framesWithError,
            framesWithObservations: framesWithObservations,
            totalObservations: totalObservations,
            totalJoints: totalJoints,
            totalOutOfRangeHighConfidence: totalOutOfRangeHighConfidence,
            totalInvalidCoordinates: totalInvalidCoordinates,
            elapsedMs: elapsedMs
        )
    }

    public func process(
        pixelBuffer: CVPixelBuffer,
        sourceFrameIndex: Int,
        processedIndex: Int,
        timestampSeconds: Double
    ) throws -> HandPoseFrameRecord {
        let handRequest = requestForFrame()
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        let requests: [VNRequest]
        switch config.requestSet {
        case .handOnly:
            requests = [handRequest]
        case .segThenHand:
            requests = [handRequest]
        case .visionBatch:
            requests = [segmentationRequest, bodyRequest, handRequest, faceRequest]
        }

        let start = DispatchTime.now().uptimeNanoseconds
        var errorRecord: VisionErrorRecord?
        do {
            if config.requestSet == .segThenHand {
                if let orientation = config.inputOrientation.cgOrientation {
                    try preflightSequenceHandler.perform(
                        [segmentationRequest],
                        on: pixelBuffer,
                        orientation: orientation
                    )
                } else {
                    try preflightSequenceHandler.perform([segmentationRequest], on: pixelBuffer)
                }
            }

            switch config.handler {
            case .image:
                let handler: VNImageRequestHandler
                if let orientation = config.inputOrientation.cgOrientation {
                    handler = VNImageRequestHandler(
                        cvPixelBuffer: pixelBuffer,
                        orientation: orientation,
                        options: [:]
                    )
                } else {
                    handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
                }
                try handler.perform(requests)
            case .sequence:
                if let orientation = config.inputOrientation.cgOrientation {
                    try sequenceHandler.perform(
                        requests,
                        on: pixelBuffer,
                        orientation: orientation
                    )
                } else {
                    try sequenceHandler.perform(requests, on: pixelBuffer)
                }
            }
        } catch {
            errorRecord = VisionErrorRecord(error)
        }

        let observations = errorRecord == nil
            ? extractHandObservations(from: handRequest, imgW: width, imgH: height)
            : []
        let frameSummary = summarize(observations)
        let durationMs = elapsedMilliseconds(since: start)

        return HandPoseFrameRecord(
            schemaVersion: 1,
            recordType: "frame",
            runId: config.runId,
            config: config,
            videoPath: videoPath,
            frameIndex: processedIndex,
            sourceFrameIndex: sourceFrameIndex,
            timestampSeconds: timestampSeconds,
            imageWidth: width,
            imageHeight: height,
            effectiveRevision: effectiveRevision,
            durationMs: durationMs,
            error: errorRecord,
            observations: observations,
            summary: frameSummary
        )
    }

    private func requestForFrame() -> VNDetectHumanHandPoseRequest {
        switch config.requestLifetime {
        case .fresh:
            return makeHandRequest()
        case .reused:
            if let reusedHandRequest { return reusedHandRequest }
            let request = makeHandRequest()
            reusedHandRequest = request
            return request
        }
    }

    private func makeHandRequest() -> VNDetectHumanHandPoseRequest {
        let request = VNDetectHumanHandPoseRequest()
        request.maximumHandCount = config.maximumHandCount
        if let revision = config.revision {
            request.revision = revision
        }
        return request
    }
}

public func makeRunName(
    requestLifetime: RequestLifetime,
    handler: HandlerKind,
    requestSet: RequestSetKind,
    usesCPUOnly: Bool,
    maximumHandCount: Int,
    orientation: VisionInputOrientation
) -> String {
    [
        requestLifetime.rawValue,
        handler.rawValue,
        requestSet.rawValue,
        usesCPUOnly ? "cpu" : "ane_default",
        "hands\(maximumHandCount)",
        "orient_\(orientation.rawValue)",
    ].joined(separator: "_")
}

func extractHandObservations(
    from request: VNDetectHumanHandPoseRequest,
    imgW: Int,
    imgH: Int
) -> [HandObservationRecord] {
    guard let observations = request.results else { return [] }
    return observations.enumerated().map { index, observation in
        let points: [VNHumanHandPoseObservation.JointName: VNRecognizedPoint]
        do {
            points = try observation.recognizedPoints(.all)
        } catch {
            return HandObservationRecord(
                index: index,
                chirality: chiralityName(observation.chirality),
                extractError: String(describing: error),
                summary: ObservationSummary(
                    jointCount: 0,
                    normalConfidenceCount: 0,
                    zeroConfidenceCount: 0,
                    outOfRangeHighConfidenceCount: 0,
                    negativeConfidenceCount: 0,
                    nonFiniteConfidenceCount: 0,
                    invalidCoordinateCount: 0,
                    minConfidence: nil,
                    maxConfidence: nil
                ),
                joints: []
            )
        }

        let joints = points
            .map { key, point in
                let normX = Double(point.location.x)
                let normY = Double(point.location.y)
                return HandJointRecord(
                    name: key.rawValue.rawValue,
                    confidence: Float(point.confidence),
                    confidenceClass: confidenceClass(Float(point.confidence)),
                    normalizedX: normX,
                    normalizedY: normY,
                    pixelX: normX * Double(imgW),
                    pixelY: (1.0 - normY) * Double(imgH),
                    coordinateClass: coordinateClass(x: normX, y: normY)
                )
            }
            .sorted { $0.name < $1.name }

        return HandObservationRecord(
            index: index,
            chirality: chiralityName(observation.chirality),
            extractError: nil,
            summary: summarize(joints),
            joints: joints
        )
    }
}

func summarize(_ observations: [HandObservationRecord]) -> FrameSummary {
    observations.reduce(
        FrameSummary(
            observationCount: observations.count,
            jointCount: 0,
            normalConfidenceCount: 0,
            zeroConfidenceCount: 0,
            outOfRangeHighConfidenceCount: 0,
            negativeConfidenceCount: 0,
            nonFiniteConfidenceCount: 0,
            invalidCoordinateCount: 0
        )
    ) { partial, observation in
        FrameSummary(
            observationCount: partial.observationCount,
            jointCount: partial.jointCount + observation.summary.jointCount,
            normalConfidenceCount: partial.normalConfidenceCount + observation.summary.normalConfidenceCount,
            zeroConfidenceCount: partial.zeroConfidenceCount + observation.summary.zeroConfidenceCount,
            outOfRangeHighConfidenceCount: partial.outOfRangeHighConfidenceCount + observation.summary.outOfRangeHighConfidenceCount,
            negativeConfidenceCount: partial.negativeConfidenceCount + observation.summary.negativeConfidenceCount,
            nonFiniteConfidenceCount: partial.nonFiniteConfidenceCount + observation.summary.nonFiniteConfidenceCount,
            invalidCoordinateCount: partial.invalidCoordinateCount + observation.summary.invalidCoordinateCount
        )
    }
}

func summarize(_ joints: [HandJointRecord]) -> ObservationSummary {
    var normal = 0
    var zero = 0
    var high = 0
    var negative = 0
    var nonFinite = 0
    var invalidCoordinates = 0
    var minConfidence: Float?
    var maxConfidence: Float?

    for joint in joints {
        switch joint.confidenceClass {
        case "normal": normal += 1
        case "zero": zero += 1
        case "out_of_range_high": high += 1
        case "negative": negative += 1
        case "non_finite": nonFinite += 1
        default: break
        }
        if joint.coordinateClass != "normal" {
            invalidCoordinates += 1
        }
        if joint.confidence.isFinite {
            minConfidence = min(minConfidence ?? joint.confidence, joint.confidence)
            maxConfidence = max(maxConfidence ?? joint.confidence, joint.confidence)
        }
    }

    return ObservationSummary(
        jointCount: joints.count,
        normalConfidenceCount: normal,
        zeroConfidenceCount: zero,
        outOfRangeHighConfidenceCount: high,
        negativeConfidenceCount: negative,
        nonFiniteConfidenceCount: nonFinite,
        invalidCoordinateCount: invalidCoordinates,
        minConfidence: minConfidence,
        maxConfidence: maxConfidence
    )
}

func confidenceClass(_ confidence: Float) -> String {
    if !confidence.isFinite { return "non_finite" }
    if confidence < 0 { return "negative" }
    if confidence == 0 { return "zero" }
    if confidence <= 1 { return "normal" }
    return "out_of_range_high"
}

func coordinateClass(x: Double, y: Double) -> String {
    if !x.isFinite || !y.isFinite { return "non_finite" }
    if x < 0 || x > 1 || y < 0 || y > 1 { return "out_of_range" }
    return "normal"
}

func chiralityName(_ chirality: VNChirality) -> String {
    switch chirality {
    case .left: return "left"
    case .right: return "right"
    default: return "unknown"
    }
}

func elapsedMilliseconds(since start: UInt64) -> Double {
    Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000.0
}

extension VisionErrorRecord {
    init(_ error: Error) {
        let nsError = error as NSError
        self.init(
            domain: nsError.domain,
            code: nsError.code,
            description: nsError.localizedDescription
        )
    }
}

struct InvestigationError: LocalizedError {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var errorDescription: String? { message }
}
