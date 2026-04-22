import AVFoundation
import CoreVideo
import Foundation
import HandPoseInvestigation
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
      SyphonVideoPublisher --video PATH [options]

    Options:
      --server-name NAME       Syphon server name. Defaults to HandPoseFixture.
      --start-seconds N        Seek before publishing. Defaults to 0.
      --duration-seconds N     Source duration per loop.
      --run-seconds N          Wall-clock duration. Defaults to one pass through the source window.
      --frame-stride N         Publish every Nth source frame. Defaults to 1.
      --loop                   Loop source frames until --run-seconds expires.
      --flipped                Publish with Syphon flipped flag set.
      --help
    """
}

func parseDouble(_ raw: String?, flag: String, default defaultValue: Double) throws -> Double {
    guard let raw else { return defaultValue }
    guard let value = Double(raw) else { throw CLIError("Invalid --\(flag) value '\(raw)'") }
    return value
}

func parseOptionalDouble(_ raw: String?, flag: String) throws -> Double? {
    guard let raw else { return nil }
    guard let value = Double(raw) else { throw CLIError("Invalid --\(flag) value '\(raw)'") }
    return value
}

func parseInt(_ raw: String?, flag: String, default defaultValue: Int) throws -> Int {
    guard let raw else { return defaultValue }
    guard let value = Int(raw) else { throw CLIError("Invalid --\(flag) value '\(raw)'") }
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

final class Publisher {
    private let videoURL: URL
    private let startSeconds: Double
    private let durationSeconds: Double?
    private let frameStride: Int
    private let shouldLoop: Bool
    private let runSeconds: Double?
    private let flipped: Bool
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let server: SyphonMetalServer
    private var textureCache: CVMetalTextureCache?
    private let wallStart = CFAbsoluteTimeGetCurrent()
    private var publishCount = 0

    init(
        videoURL: URL,
        serverName: String,
        startSeconds: Double,
        durationSeconds: Double?,
        frameStride: Int,
        shouldLoop: Bool,
        runSeconds: Double?,
        flipped: Bool
    ) throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else {
            throw CLIError("Metal device/command queue unavailable")
        }
        let server = SyphonMetalServer(name: serverName, device: device, options: nil)

        self.videoURL = videoURL
        self.startSeconds = startSeconds
        self.durationSeconds = durationSeconds
        self.frameStride = frameStride
        self.shouldLoop = shouldLoop
        self.runSeconds = runSeconds
        self.flipped = flipped
        self.device = device
        self.commandQueue = commandQueue
        self.server = server

        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
    }

    func run() throws {
        printErr("Syphon server '\(server.name ?? "unnamed")' publishing \(videoURL.path)")
        repeat {
            var previousTimestamp: Double?
            try VideoFrameReader.read(
                videoURL: videoURL,
                startSeconds: startSeconds,
                durationSeconds: durationSeconds,
                maxFrames: nil,
                frameStride: frameStride
            ) { frame in
                if shouldStop() {
                    throw StopPublishing()
                }
                pace(frameTimestamp: frame.timestampSeconds, previousTimestamp: &previousTimestamp)
                try publish(frame.pixelBuffer)
                publishCount += 1
                if publishCount % 120 == 0 {
                    printErr("published \(publishCount) frames")
                }
                RunLoop.current.run(mode: .default, before: Date())
            }
        } while shouldLoop && !shouldStop()

        printErr("published \(publishCount) frames total")
    }

    private func shouldStop() -> Bool {
        guard let runSeconds else { return false }
        return CFAbsoluteTimeGetCurrent() - wallStart >= runSeconds
    }

    private func pace(frameTimestamp: Double, previousTimestamp: inout Double?) {
        defer { previousTimestamp = frameTimestamp }
        guard let previousTimestamp else { return }
        let delta = max(0.0, min(0.25, frameTimestamp - previousTimestamp))
        if delta > 0 {
            Thread.sleep(forTimeInterval: delta)
        }
    }

    private func publish(_ pixelBuffer: CVPixelBuffer) throws {
        guard let textureCache else { throw CLIError("Missing CVMetalTextureCache") }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            textureCache,
            pixelBuffer,
            nil,
            .bgra8Unorm,
            width,
            height,
            0,
            &cvTexture
        )
        guard status == kCVReturnSuccess,
              let cvTexture,
              let texture = CVMetalTextureGetTexture(cvTexture),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw CLIError("Could not create Metal texture from video frame")
        }

        server.publishFrameTexture(
            texture,
            on: commandBuffer,
            imageRegion: CGRect(x: 0, y: 0, width: width, height: height),
            flipped: flipped
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
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

    let startSeconds = try parseDouble(cli.value("start-seconds"), flag: "start-seconds", default: 0)
    let durationSeconds = try parseOptionalDouble(cli.value("duration-seconds"), flag: "duration-seconds")
    let frameStride = try parseInt(cli.value("frame-stride"), flag: "frame-stride", default: 1)
    let runSeconds = try parseOptionalDouble(cli.value("run-seconds"), flag: "run-seconds")
    guard startSeconds >= 0 else { throw CLIError("--start-seconds must be >= 0") }
    if let durationSeconds, durationSeconds <= 0 { throw CLIError("--duration-seconds must be > 0") }
    if let runSeconds, runSeconds <= 0 { throw CLIError("--run-seconds must be > 0") }
    guard frameStride > 0 else { throw CLIError("--frame-stride must be >= 1") }

    let publisher = try Publisher(
        videoURL: fileURL(videoPath),
        serverName: cli.value("server-name") ?? "HandPoseFixture",
        startSeconds: startSeconds,
        durationSeconds: durationSeconds,
        frameStride: frameStride,
        shouldLoop: cli.has("loop"),
        runSeconds: runSeconds,
        flipped: cli.has("flipped")
    )
    try publisher.run()
} catch is StopPublishing {
    printErr("publish duration reached")
} catch {
    printErr("error: \(error.localizedDescription)")
    exit(1)
}

struct StopPublishing: Error {}

struct CLIError: LocalizedError {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var errorDescription: String? { message }
}
