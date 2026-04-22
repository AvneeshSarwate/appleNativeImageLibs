import Foundation
import HandPoseInvestigation

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

    func bool(_ flag: String, default defaultValue: Bool) -> Bool {
        guard has(flag) else { return defaultValue }
        guard let raw = value(flag) else { return true }
        switch raw.lowercased() {
        case "1", "true", "yes", "y", "on": return true
        case "0", "false", "no", "n", "off": return false
        default: return true
        }
    }
}

func usage() -> String {
    """
    Usage:
      HandPoseReplay --video PATH [options]

    Options:
      --output PATH              JSONL output path. Defaults to stdout.
      --start-seconds N          Seek to this timestamp before sampling frames. Defaults to 0.
      --duration-seconds N       Stop after this many seconds of source video.
      --max-frames N            Stop after N processed frames.
      --frame-stride N          Process every Nth source frame. Defaults to 1.
      --request fresh|reused    Hand request lifetime. Defaults to fresh.
      --handler image|sequence  Vision handler. Defaults to image.
      --request-set hand_only|seg_then_hand|vision_batch
                                 Run hand alone, VisionApp-style seg then hand, or all requests batched.
                                 Defaults to hand_only.
      --max-hands N             VNDetectHumanHandPoseRequest.maximumHandCount. Defaults to 2.
      --revision N              Pin VNDetectHumanHandPoseRequest.revision.
      --orientation unspecified|up|down|left|right
                                 Pixel-buffer orientation passed to Vision. Defaults to unspecified,
                                 matching VisionApp's no-orientation handler overload.
      --run-id ID               Stable run id for JSONL grouping. Defaults to UUID.
      --name NAME               Human readable config name. Defaults to generated name.
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

func parseInt(_ raw: String?, flag: String, default defaultValue: Int) throws -> Int {
    guard let raw else { return defaultValue }
    guard let value = Int(raw) else {
        throw CLIError("Invalid --\(flag) value '\(raw)'")
    }
    return value
}

func parseOptionalInt(_ raw: String?, flag: String) throws -> Int? {
    guard let raw else { return nil }
    guard let value = Int(raw) else {
        throw CLIError("Invalid --\(flag) value '\(raw)'")
    }
    return value
}

func parseDouble(_ raw: String?, flag: String, default defaultValue: Double) throws -> Double {
    guard let raw else { return defaultValue }
    guard let value = Double(raw) else {
        throw CLIError("Invalid --\(flag) value '\(raw)'")
    }
    return value
}

func parseOptionalDouble(_ raw: String?, flag: String) throws -> Double? {
    guard let raw else { return nil }
    guard let value = Double(raw) else {
        throw CLIError("Invalid --\(flag) value '\(raw)'")
    }
    return value
}

func fileURL(_ path: String) -> URL {
    if path.hasPrefix("/") {
        return URL(fileURLWithPath: path).standardizedFileURL
    }
    return URL(
        fileURLWithPath: path,
        relativeTo: URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    ).standardizedFileURL
}

func printErr(_ text: String) {
    FileHandle.standardError.write(Data((text + "\n").utf8))
}

let cli = CLI(args: Array(CommandLine.arguments.dropFirst()))

do {
    if cli.has("help") {
        print(usage())
        exit(0)
    }

    guard let videoPath = cli.value("video") else {
        throw CLIError("Missing --video PATH\n\n\(usage())")
    }

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
        default: .handOnly
    )
    let orientation = try parseEnum(
        VisionInputOrientation.self,
        raw: cli.value("orientation"),
        flag: "orientation",
        default: .unspecified
    )
    let maxFrames = try parseOptionalInt(cli.value("max-frames"), flag: "max-frames")
    let startSeconds = try parseDouble(cli.value("start-seconds"), flag: "start-seconds", default: 0)
    let durationSeconds = try parseOptionalDouble(cli.value("duration-seconds"), flag: "duration-seconds")
    let frameStride = try parseInt(cli.value("frame-stride"), flag: "frame-stride", default: 1)
    let maximumHandCount = try parseInt(cli.value("max-hands"), flag: "max-hands", default: 2)
    let revision = try parseOptionalInt(cli.value("revision"), flag: "revision")
    if cli.has("cpu-only") {
        throw CLIError("--cpu-only is intentionally unsupported; this tool tests Vision's default Apple compute path")
    }
    let usesCPUOnly = false
    guard startSeconds >= 0 else { throw CLIError("--start-seconds must be >= 0") }
    if let durationSeconds, durationSeconds <= 0 {
        throw CLIError("--duration-seconds must be > 0")
    }
    guard frameStride > 0 else { throw CLIError("--frame-stride must be >= 1") }
    guard maximumHandCount > 0 else { throw CLIError("--max-hands must be >= 1") }

    let generatedName = makeRunName(
        requestLifetime: requestLifetime,
        handler: handler,
        requestSet: requestSet,
        usesCPUOnly: usesCPUOnly,
        maximumHandCount: maximumHandCount,
        orientation: orientation
    )
    let config = HandPoseRunConfig(
        runId: cli.value("run-id") ?? UUID().uuidString,
        name: cli.value("name") ?? generatedName,
        requestLifetime: requestLifetime,
        handler: handler,
        requestSet: requestSet,
        usesCPUOnly: usesCPUOnly,
        maximumHandCount: maximumHandCount,
        revision: revision,
        inputOrientation: orientation,
        startSeconds: startSeconds,
        durationSeconds: durationSeconds,
        maxFrames: maxFrames,
        frameStride: frameStride
    )

    let videoURL = fileURL(videoPath)
    let writer = try JSONLineWriter(path: cli.value("output"))
    let runner = try HandPoseReplayRunner(config: config, videoPath: videoURL.path)
    let summary = try runner.run(videoURL: videoURL, writer: writer)

    printErr(
        "run_id=\(summary.runId) frames=\(summary.framesProcessed) " +
            "errors=\(summary.framesWithError) observations=\(summary.totalObservations) " +
            "out_of_range_conf=\(summary.totalOutOfRangeHighConfidence) " +
            String(format: "elapsed_ms=%.1f", summary.elapsedMs)
    )
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
