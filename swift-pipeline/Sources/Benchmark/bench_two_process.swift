import Foundation
import CoreML
import CoreVideo

/// RTMPose worker: loads model, reads bbox from stdin, runs inference, writes ack to stdout.
func runRTMWorker() async throws {
    let dir = FileManager.default.currentDirectoryPath
    let compiled = try await MLModel.compileModel(
        at: URL(fileURLWithPath: "\(dir)/rtmpose.mlpackage"))
    let cfg = MLModelConfiguration()
    cfg.computeUnits = .all
    let model = try MLModel(contentsOf: compiled, configuration: cfg)
    let inputName = model.modelDescription.inputDescriptionsByName.keys.first!

    // Preallocate input
    let arr = try MLMultiArray(shape: [1, 3, 384, 288], dataType: .float32)
    let input = try MLDictionaryFeatureProvider(dictionary: [
        inputName: MLFeatureValue(multiArray: arr)
    ])

    // Warmup
    for _ in 0..<10 { let _ = try await model.prediction(from: input) }

    // Signal ready
    let readyByte: [UInt8] = [0x01]
    FileHandle.standardOutput.write(Data(readyByte))

    // Process loop: read 1 byte trigger, run inference, write 1 byte + timing
    let stdin = FileHandle.standardInput
    while true {
        let trigger = stdin.readData(ofLength: 1)
        if trigger.isEmpty { break }

        let t0 = CFAbsoluteTimeGetCurrent()
        let _ = try await model.prediction(from: input)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        // Write 8 bytes (Double) timing back
        var time = elapsed
        let timeData = Data(bytes: &time, count: 8)
        FileHandle.standardOutput.write(timeData)
    }
}

/// Two-process benchmark: parent runs YOLO, child runs RTMPose in separate process.
func runTwoProcessBench() async throws {
    let dir = FileManager.default.currentDirectoryPath
    let execPath = CommandLine.arguments[0]

    // --- Load YOLO in parent ---
    let yoloCompiled = try await MLModel.compileModel(
        at: URL(fileURLWithPath: "\(dir)/yolo11s-seg.mlpackage"))
    let yCfg = MLModelConfiguration()
    yCfg.computeUnits = .all
    let yoloModel = try MLModel(contentsOf: yoloCompiled, configuration: yCfg)
    let yoloIn = yoloModel.modelDescription.inputDescriptionsByName.keys.first!

    let pb = makePixelBuffer(640, 640)
    CVPixelBufferLockBaseAddress(pb, [])
    memset(CVPixelBufferGetBaseAddress(pb)!, 128, CVPixelBufferGetBytesPerRow(pb) * 640)
    CVPixelBufferUnlockBaseAddress(pb, [])

    let yoloFP = try MLDictionaryFeatureProvider(dictionary: [
        yoloIn: MLFeatureValue(pixelBuffer: pb)
    ])

    // Warmup YOLO
    for _ in 0..<10 { let _ = try await yoloModel.prediction(from: yoloFP) }

    // --- Spawn RTMPose worker process ---
    print("Spawning RTMPose worker process...")
    let worker = Process()
    worker.executableURL = URL(fileURLWithPath: execPath)
    worker.arguments = ["--rtm-worker"]
    worker.currentDirectoryURL = URL(fileURLWithPath: dir)

    let toWorker = Pipe()    // parent writes, worker reads on stdin
    let fromWorker = Pipe()  // worker writes, parent reads on stdout
    worker.standardInput = toWorker
    worker.standardOutput = fromWorker
    // Suppress worker's stderr
    worker.standardError = FileHandle.nullDevice

    try worker.run()

    let workerOut = fromWorker.fileHandleForReading
    let workerIn = toWorker.fileHandleForWriting

    // Wait for worker ready signal (1 byte)
    let ready = workerOut.readData(ofLength: 1)
    guard ready.count == 1 else {
        print("Worker failed to start")
        return
    }
    print("Worker ready.\n")

    // --- Benchmark: single-process alternating (baseline) ---
    print("=== Single-process alternating (200 iters) ===\n")

    // Load RTMPose in parent too for comparison
    let poseCompiled = try await MLModel.compileModel(
        at: URL(fileURLWithPath: "\(dir)/rtmpose.mlpackage"))
    let pCfg = MLModelConfiguration()
    pCfg.computeUnits = .all
    let poseModel = try MLModel(contentsOf: poseCompiled, configuration: pCfg)
    let poseIn = poseModel.modelDescription.inputDescriptionsByName.keys.first!
    let poseArr = try MLMultiArray(shape: [1, 3, 384, 288], dataType: .float32)
    let poseFP = try MLDictionaryFeatureProvider(dictionary: [
        poseIn: MLFeatureValue(multiArray: poseArr)
    ])
    for _ in 0..<10 { let _ = try await poseModel.prediction(from: poseFP) }

    var spYT = [Double](), spPT = [Double]()
    for _ in 0..<200 {
        let t0 = CFAbsoluteTimeGetCurrent()
        let _ = try await yoloModel.prediction(from: yoloFP)
        let t1 = CFAbsoluteTimeGetCurrent()
        let _ = try await poseModel.prediction(from: poseFP)
        let t2 = CFAbsoluteTimeGetCurrent()
        spYT.append(t1 - t0); spPT.append(t2 - t1)
    }
    let spym = median(spYT) * 1000, sppm = median(spPT) * 1000
    print("  YOLO:    \(String(format: "%6.2f", spym))ms")
    print("  RTMPose: \(String(format: "%6.2f", sppm))ms")
    print("  Total:   \(String(format: "%6.2f", spym + sppm))ms")

    // --- Benchmark: two-process sequential (200 iters) ---
    print("\n=== Two-process sequential (200 iters) ===")
    print("  (YOLO in parent, RTMPose in child process)\n")

    let trigger: [UInt8] = [0x01]
    var tpYT = [Double](), tpPT = [Double](), tpTotal = [Double]()

    for _ in 0..<200 {
        let t0 = CFAbsoluteTimeGetCurrent()

        // YOLO in parent
        let _ = try await yoloModel.prediction(from: yoloFP)
        let t1 = CFAbsoluteTimeGetCurrent()

        // Trigger RTMPose in child
        workerIn.write(Data(trigger))

        // Read timing from child (8 bytes Double)
        let timeData = workerOut.readData(ofLength: 8)
        let t2 = CFAbsoluteTimeGetCurrent()

        var childTime: Double = 0
        if timeData.count == 8 {
            _ = withUnsafeMutableBytes(of: &childTime) { ptr in
                timeData.copyBytes(to: ptr)
            }
        }

        tpYT.append(t1 - t0)
        tpPT.append(childTime)
        tpTotal.append(t2 - t0)
    }
    let tpym = median(tpYT) * 1000
    let tppm = median(tpPT) * 1000
    let tptm = median(tpTotal) * 1000
    print("  YOLO (parent):      \(String(format: "%6.2f", tpym))ms")
    print("  RTMPose (child):    \(String(format: "%6.2f", tppm))ms  ← compare to \(String(format: "%.2f", sppm))ms single-process")
    print("  Total (wall clock): \(String(format: "%6.2f", tptm))ms  (includes IPC)")
    print("  IPC overhead:       \(String(format: "%6.2f", tptm - tpym - tppm))ms")

    // --- Summary ---
    print("\n=== Summary ===\n")
    print("  Single-process alternating: \(String(format: "%.1f", spym + sppm))ms")
    print("  Two-process sequential:     \(String(format: "%.1f", tptm))ms")
    let saved = (spym + sppm) - tptm
    print("  Savings:                    \(String(format: "%+.1f", saved))ms")
    if tppm < sppm {
        print("  RTMPose penalty eliminated: \(String(format: "%.1f", sppm))ms → \(String(format: "%.1f", tppm))ms " +
              "(\(String(format: "%.1f", sppm - tppm))ms saved)")
    } else {
        print("  RTMPose penalty NOT eliminated")
    }

    // Cleanup
    workerIn.closeFile()
    worker.terminate()
    worker.waitUntilExit()
}
