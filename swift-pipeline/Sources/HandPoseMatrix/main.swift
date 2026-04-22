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
}

struct MatrixSummaryFile: Codable {
    let schemaVersion: Int
    let recordType: String
    let generatedAt: String
    let videoPath: String
    let jsonlPath: String
    let summaries: [HandPoseRunSummary]
}

func usage() -> String {
    """
    Usage:
      HandPoseMatrix --video PATH [options]

    Options:
      --output-dir PATH          Output directory. Defaults to ./hand_pose_matrix_<timestamp>.
      --start-seconds N          Seek to this timestamp before sampling frames. Defaults to 0.
      --duration-seconds N       Stop each run after this many seconds of source video.
      --max-frames N            Stop each run after N processed frames.
      --frame-stride N          Process every Nth source frame. Defaults to 1.
      --requests fresh,reused   Request lifetimes to test. Defaults to fresh,reused.
      --handlers image,sequence Vision handlers to test. Defaults to image,sequence.
      --request-sets hand_only,seg_then_hand,vision_batch
                                Request sets to test. Defaults to all three.
      --max-hands-values 1,2    maximumHandCount values to test. Defaults to 1,2.
      --revision N              Pin VNDetectHumanHandPoseRequest.revision for every run.
      --orientation unspecified|up|down|left|right
                                Pixel-buffer orientation passed to Vision. Defaults to unspecified,
                                matching VisionApp's no-orientation handler overload.
      --help
    """
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

func parseBoolCSV(_ raw: String?, flag: String, default defaultValue: [Bool]) throws -> [Bool] {
    guard let values = csv(raw) else { return defaultValue }
    return try values.map { rawValue in
        switch rawValue.lowercased() {
        case "1", "true", "yes", "y", "on": return true
        case "0", "false", "no", "n", "off": return false
        default: throw CLIError("Invalid --\(flag) value '\(rawValue)'")
        }
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

func parseEnum<T: RawRepresentable>(_ type: T.Type, raw: String?, flag: String, default defaultValue: T) throws -> T
where T.RawValue == String {
    guard let raw else { return defaultValue }
    guard let value = T(rawValue: raw) else {
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

func parseInt(_ raw: String?, flag: String, default defaultValue: Int) throws -> Int {
    guard let raw else { return defaultValue }
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

func timestamp() -> String {
    let formatter = DateFormatter()
    formatter.dateFormat = "yyyyMMdd_HHmmss"
    return formatter.string(from: Date())
}

func isoTimestamp() -> String {
    ISO8601DateFormatter().string(from: Date())
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
    if cli.value("cpu-only-values") != nil {
        throw CLIError("--cpu-only-values is intentionally unsupported; this matrix tests Vision's default Apple compute path")
    }
    let cpuOnlyValues = [false]
    let maxHandValues = try parseIntCSV(
        cli.value("max-hands-values"),
        flag: "max-hands-values",
        default: [1, 2]
    )
    let orientation = try parseEnum(
        VisionInputOrientation.self,
        raw: cli.value("orientation"),
        flag: "orientation",
        default: .unspecified
    )
    let startSeconds = try parseDouble(cli.value("start-seconds"), flag: "start-seconds", default: 0)
    let durationSeconds = try parseOptionalDouble(cli.value("duration-seconds"), flag: "duration-seconds")
    let maxFrames = try parseOptionalInt(cli.value("max-frames"), flag: "max-frames")
    let frameStride = try parseInt(cli.value("frame-stride"), flag: "frame-stride", default: 1)
    let revision = try parseOptionalInt(cli.value("revision"), flag: "revision")
    guard startSeconds >= 0 else { throw CLIError("--start-seconds must be >= 0") }
    if let durationSeconds, durationSeconds <= 0 {
        throw CLIError("--duration-seconds must be > 0")
    }
    guard frameStride > 0 else { throw CLIError("--frame-stride must be >= 1") }
    guard maxHandValues.allSatisfy({ $0 > 0 }) else {
        throw CLIError("--max-hands-values entries must be >= 1")
    }

    let videoURL = fileURL(videoPath)
    let outputDirPath = cli.value("output-dir") ?? "hand_pose_matrix_\(timestamp())"
    let outputDirURL = fileURL(outputDirPath)
    try FileManager.default.createDirectory(at: outputDirURL, withIntermediateDirectories: true)

    let jsonlPath = outputDirURL.appendingPathComponent("matrix.jsonl").path
    let summaryPath = outputDirURL.appendingPathComponent("summary.json").path
    let writer = try JSONLineWriter(path: jsonlPath)
    var summaries: [HandPoseRunSummary] = []
    var runIndex = 0

    for requestLifetime in requestLifetimes {
        for handler in handlers {
            for requestSet in requestSets {
                for usesCPUOnly in cpuOnlyValues {
                    for maximumHandCount in maxHandValues {
                        let runName = makeRunName(
                            requestLifetime: requestLifetime,
                            handler: handler,
                            requestSet: requestSet,
                            usesCPUOnly: usesCPUOnly,
                            maximumHandCount: maximumHandCount,
                            orientation: orientation
                        )
                        let config = HandPoseRunConfig(
                            runId: String(format: "%03d_%@", runIndex, runName),
                            name: runName,
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

                        printErr("[\(runIndex + 1)] \(runName)")
                        let runner = try HandPoseReplayRunner(config: config, videoPath: videoURL.path)
                        let summary = try runner.run(videoURL: videoURL, writer: writer)
                        summaries.append(summary)
                        printErr(
                            "    frames=\(summary.framesProcessed) errors=\(summary.framesWithError) " +
                                "obs=\(summary.totalObservations) " +
                                "out_of_range_conf=\(summary.totalOutOfRangeHighConfidence)"
                        )
                        runIndex += 1
                    }
                }
            }
        }
    }

    let summaryFile = MatrixSummaryFile(
        schemaVersion: 1,
        recordType: "matrix_summary",
        generatedAt: isoTimestamp(),
        videoPath: videoURL.path,
        jsonlPath: jsonlPath,
        summaries: summaries
    )
    try JSONFileWriter().write(summaryFile, to: summaryPath)
    printErr("wrote \(jsonlPath)")
    printErr("wrote \(summaryPath)")
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
