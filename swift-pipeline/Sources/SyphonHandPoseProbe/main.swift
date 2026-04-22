import CoreVideo
import Foundation
import HandPoseInvestigation
import IOSurface
import Metal
import Syphon

struct CLI {
    let args: [String]

    func value(_ flag: String) -> String? {
        guard let index = args.firstIndex(of: "--\(flag)"),
              index + 1 < args.count,
              !args[index + 1].hasPrefix("--") else {
            return nil
        }
        return args[index + 1]
    }

    func has(_ flag: String) -> Bool {
        args.contains("--\(flag)")
    }
}

func usage() -> String {
    """
    Usage:
      SyphonHandPoseProbe [options]

    Options:
      --server-name NAME           Syphon server name. Defaults to HandPoseFixture.
      --app-name NAME              Optional Syphon publisher app-name filter.
      --output PATH                JSONL output path. Defaults to stdout.
      --duration-seconds N         Wall-clock capture duration. Defaults to 30.
      --max-frames N               Stop after N received frames.
      --discovery-timeout N        Server discovery timeout. Defaults to 10.
      --grid                       Run the full request/handler/request-set/max-hands grid.
      --request fresh|reused       Hand request lifetime. Defaults to fresh.
      --requests fresh,reused      Grid request lifetimes. Defaults to fresh,reused.
      --handler image|sequence     Vision handler. Defaults to image.
      --handlers image,sequence    Grid handlers. Defaults to image,sequence.
      --request-set hand_only|seg_then_hand|vision_batch
                                  Defaults to seg_then_hand.
      --request-sets hand_only,seg_then_hand,vision_batch
                                  Grid request sets. Defaults to all three.
      --max-hands N                maximumHandCount. Defaults to 2.
      --max-hands-values 1,2       Grid maximumHandCount values. Defaults to 1,2.
      --orientation unspecified|up|down|left|right
                                  Defaults to unspecified.
      --help
    """
}

func parseEnum<T: RawRepresentable>(_ type: T.Type, raw: String?, flag: String, default defaultValue: T) throws -> T
where T.RawValue == String {
    guard let raw else { return defaultValue }
    guard let value = T(rawValue: raw) else {
        throw CLIError("Invalid --\(flag) value '\(raw)'")
    }
    return value
}

func csv(_ raw: String?) -> [String]? {
    raw?.split(separator: ",").map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty }
}

func parseCSVEnum<T: RawRepresentable>(_ type: T.Type, raw: String?, flag: String, default defaultValue: [T]) throws -> [T]
where T.RawValue == String {
    guard let values = csv(raw) else { return defaultValue }
    return try values.map { rawValue in
        guard let value = T(rawValue: rawValue) else {
            throw CLIError("Invalid --\(flag) value '\(rawValue)'")
        }
        return value
    }
}

func parseIntCSV(_ raw: String?, flag: String, default defaultValue: [Int]) throws -> [Int] {
    guard let values = csv(raw) else { return defaultValue }
    return try values.map { rawValue in
        guard let value = Int(rawValue) else {
            throw CLIError("Invalid --\(flag) value '\(rawValue)'")
        }
        return value
    }
}

func parseDouble(_ raw: String?, flag: String, default defaultValue: Double) throws -> Double {
    guard let raw else { return defaultValue }
    guard let value = Double(raw) else { throw CLIError("Invalid --\(flag) value '\(raw)'") }
    return value
}

func parseOptionalInt(_ raw: String?, flag: String) throws -> Int? {
    guard let raw else { return nil }
    guard let value = Int(raw) else { throw CLIError("Invalid --\(flag) value '\(raw)'") }
    return value
}

func parseInt(_ raw: String?, flag: String, default defaultValue: Int) throws -> Int {
    guard let raw else { return defaultValue }
    guard let value = Int(raw) else { throw CLIError("Invalid --\(flag) value '\(raw)'") }
    return value
}

func printErr(_ text: String) {
    FileHandle.standardError.write(Data((text + "\n").utf8))
}

final class TextureConverter {
    func pixelBuffer(from texture: MTLTexture) throws -> CVPixelBuffer {
        if let surface = texture.iosurface {
            var unmanagedPixelBuffer: Unmanaged<CVPixelBuffer>?
            let attrs: [String: Any] = [
                kCVPixelBufferMetalCompatibilityKey as String: true,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
            ]
            let status = CVPixelBufferCreateWithIOSurface(
                kCFAllocatorDefault,
                surface,
                attrs as CFDictionary,
                &unmanagedPixelBuffer
            )
            if status == kCVReturnSuccess, let unmanagedPixelBuffer {
                return unmanagedPixelBuffer.takeRetainedValue()
            }
        }

        var pixelBuffer: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            texture.width,
            texture.height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let pixelBuffer else {
            throw CLIError("Could not allocate CVPixelBuffer for Syphon texture")
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw CLIError("CVPixelBuffer has no base address")
        }
        texture.getBytes(
            base,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
            from: MTLRegionMake2D(0, 0, texture.width, texture.height),
            mipmapLevel: 0
        )
        return pixelBuffer
    }
}

final class Probe {
    private let serverName: String
    private let appName: String?
    private let durationSeconds: Double
    private let maxFrames: Int?
    private let discoveryTimeout: Double
    private let writer: JSONLineWriter
    private let converter = TextureConverter()
    private let device: MTLDevice
    private let frameSemaphore = DispatchSemaphore(value: 0)
    private var client: SyphonMetalClient?
    private let configs: [HandPoseRunConfig]
    private var runners: [String: HandPoseReplayRunner] = [:]

    init(
        serverName: String,
        appName: String?,
        durationSeconds: Double,
        maxFrames: Int?,
        discoveryTimeout: Double,
        writer: JSONLineWriter,
        configs: [HandPoseRunConfig]
    ) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CLIError("Metal device unavailable")
        }
        self.serverName = serverName
        self.appName = appName
        self.durationSeconds = durationSeconds
        self.maxFrames = maxFrames
        self.discoveryTimeout = discoveryTimeout
        self.writer = writer
        self.configs = configs
        self.device = device
    }

    func run() throws {
        let description = try waitForServer()
        printErr("connected to Syphon server: \(description)")

        let client = SyphonMetalClient(
            serverDescription: description,
            device: device,
            options: nil
        ) { [weak self] _ in
            self?.frameSemaphore.signal()
        }
        guard client.isValid else {
            throw CLIError("Could not connect to Syphon server \(serverName)")
        }
        self.client = client
        defer { client.stop() }

        let start = CFAbsoluteTimeGetCurrent()
        var frameIndex = 0
        var errorsByRun: [String: Int] = [:]
        var highConfidenceByRun: [String: Int] = [:]
        for config in configs {
            runners[config.runId] = try HandPoseReplayRunner(config: config, videoPath: "syphon://\(serverName)")
            errorsByRun[config.runId] = 0
            highConfidenceByRun[config.runId] = 0
        }

        while CFAbsoluteTimeGetCurrent() - start < durationSeconds {
            if let maxFrames, frameIndex >= maxFrames { break }
            _ = frameSemaphore.wait(timeout: .now() + 1.0)
            RunLoop.current.run(mode: .default, before: Date())
            guard client.hasNewFrame, let texture = client.newFrameImage() else { continue }

            let pixelBuffer = try converter.pixelBuffer(from: texture)
            let timestamp = CFAbsoluteTimeGetCurrent() - start
            for config in configs {
                guard let runner = runners[config.runId] else { continue }
                let record = try runner.process(
                    pixelBuffer: pixelBuffer,
                    sourceFrameIndex: frameIndex,
                    processedIndex: frameIndex,
                    timestampSeconds: timestamp
                )
                try writer.write(record)
                if record.error != nil { errorsByRun[config.runId, default: 0] += 1 }
                highConfidenceByRun[config.runId, default: 0] += record.summary.outOfRangeHighConfidenceCount
            }
            frameIndex += 1

            if frameIndex % 120 == 0 {
                let errors = errorsByRun.values.reduce(0, +)
                let highConfidence = highConfidenceByRun.values.reduce(0, +)
                printErr(
                    "processed \(frameIndex) syphon frames across \(configs.count) configs " +
                        "errors=\(errors) high_conf=\(highConfidence)"
                )
            }
        }

        let errors = errorsByRun.values.reduce(0, +)
        let highConfidence = highConfidenceByRun.values.reduce(0, +)
        printErr(
            "processed \(frameIndex) syphon frames across \(configs.count) configs " +
                "errors=\(errors) high_conf=\(highConfidence)"
        )
    }

    private func waitForServer() throws -> [String: Any] {
        let deadline = Date().addingTimeInterval(discoveryTimeout)
        let directory = SyphonServerDirectory.shared()
        while Date() < deadline {
            let matches = directory.servers(matchingName: serverName, appName: appName)
            if let description = matches.first {
                return description
            }
            RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.1))
        }
        throw CLIError("Timed out waiting for Syphon server named \(serverName)")
    }
}

let cli = CLI(args: Array(CommandLine.arguments.dropFirst()))

do {
    if cli.has("help") {
        print(usage())
        exit(0)
    }

    let orientation = try parseEnum(
        VisionInputOrientation.self,
        raw: cli.value("orientation"),
        flag: "orientation",
        default: .unspecified
    )
    let durationSeconds = try parseDouble(cli.value("duration-seconds"), flag: "duration-seconds", default: 30)
    let discoveryTimeout = try parseDouble(cli.value("discovery-timeout"), flag: "discovery-timeout", default: 10)
    let maximumHandCount = try parseInt(cli.value("max-hands"), flag: "max-hands", default: 2)
    let maximumHandCounts = try parseIntCSV(
        cli.value("max-hands-values"),
        flag: "max-hands-values",
        default: [1, 2]
    )
    let maxFrames = try parseOptionalInt(cli.value("max-frames"), flag: "max-frames")
    guard durationSeconds > 0 else { throw CLIError("--duration-seconds must be > 0") }
    guard discoveryTimeout > 0 else { throw CLIError("--discovery-timeout must be > 0") }
    guard maximumHandCount > 0 else { throw CLIError("--max-hands must be >= 1") }
    guard maximumHandCounts.allSatisfy({ $0 > 0 }) else {
        throw CLIError("--max-hands-values entries must be >= 1")
    }

    let serverName = cli.value("server-name") ?? "HandPoseFixture"
    let configs: [HandPoseRunConfig]
    if cli.has("grid") {
        let requestLifetimes = try parseCSVEnum(
            RequestLifetime.self,
            raw: cli.value("requests"),
            flag: "requests",
            default: RequestLifetime.allCases
        )
        let handlers = try parseCSVEnum(
            HandlerKind.self,
            raw: cli.value("handlers"),
            flag: "handlers",
            default: HandlerKind.allCases
        )
        let requestSets = try parseCSVEnum(
            RequestSetKind.self,
            raw: cli.value("request-sets"),
            flag: "request-sets",
            default: RequestSetKind.allCases
        )
        var built: [HandPoseRunConfig] = []
        var runIndex = 0
        for requestLifetime in requestLifetimes {
            for handler in handlers {
                for requestSet in requestSets {
                    for maximumHandCount in maximumHandCounts {
                        let name = makeRunName(
                            requestLifetime: requestLifetime,
                            handler: handler,
                            requestSet: requestSet,
                            usesCPUOnly: false,
                            maximumHandCount: maximumHandCount,
                            orientation: orientation
                        )
                        built.append(
                            HandPoseRunConfig(
                                runId: String(format: "%03d_%@", runIndex, name),
                                name: name,
                                requestLifetime: requestLifetime,
                                handler: handler,
                                requestSet: requestSet,
                                usesCPUOnly: false,
                                maximumHandCount: maximumHandCount,
                                revision: nil,
                                inputOrientation: orientation,
                                startSeconds: 0,
                                durationSeconds: nil,
                                maxFrames: maxFrames,
                                frameStride: 1
                            )
                        )
                        runIndex += 1
                    }
                }
            }
        }
        configs = built
    } else {
        let requestLifetime = try parseEnum(
            RequestLifetime.self,
            raw: cli.value("request"),
            flag: "request",
            default: .fresh
        )
        let handler = try parseEnum(
            HandlerKind.self,
            raw: cli.value("handler"),
            flag: "handler",
            default: .image
        )
        let requestSet = try parseEnum(
            RequestSetKind.self,
            raw: cli.value("request-set"),
            flag: "request-set",
            default: .segThenHand
        )
        let name = makeRunName(
            requestLifetime: requestLifetime,
            handler: handler,
            requestSet: requestSet,
            usesCPUOnly: false,
            maximumHandCount: maximumHandCount,
            orientation: orientation
        )
        configs = [
            HandPoseRunConfig(
                runId: UUID().uuidString,
                name: name,
                requestLifetime: requestLifetime,
                handler: handler,
                requestSet: requestSet,
                usesCPUOnly: false,
                maximumHandCount: maximumHandCount,
                revision: nil,
                inputOrientation: orientation,
                startSeconds: 0,
                durationSeconds: nil,
                maxFrames: maxFrames,
                frameStride: 1
            )
        ]
    }
    printErr("configured \(configs.count) Syphon probe config(s)")

    let probe = try Probe(
        serverName: serverName,
        appName: cli.value("app-name"),
        durationSeconds: durationSeconds,
        maxFrames: maxFrames,
        discoveryTimeout: discoveryTimeout,
        writer: try JSONLineWriter(path: cli.value("output")),
        configs: configs
    )
    try probe.run()
} catch {
    printErr("error: \(error.localizedDescription)")
    exit(1)
}

struct CLIError: LocalizedError {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var errorDescription: String? { message }
}
