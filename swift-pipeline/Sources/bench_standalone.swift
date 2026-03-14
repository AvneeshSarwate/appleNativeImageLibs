import Foundation
import CoreML
import CoreVideo

func runStandaloneBench() async throws {
    let dir = FileManager.default.currentDirectoryPath

    func loadModel(_ name: String, _ units: MLComputeUnits) async throws -> MLModel {
        let compiled = try await MLModel.compileModel(at: URL(fileURLWithPath: "\(dir)/\(name)"))
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units
        return try MLModel(contentsOf: compiled, configuration: cfg)
    }

    // Prepare YOLO input
    let pb = makePixelBuffer(640, 640)
    CVPixelBufferLockBaseAddress(pb, [])
    memset(CVPixelBufferGetBaseAddress(pb)!, 128, CVPixelBufferGetBytesPerRow(pb) * 640)
    CVPixelBufferUnlockBaseAddress(pb, [])

    // Prepare RTMPose input
    let poseArr = try MLMultiArray(shape: [1, 3, 384, 288], dataType: .float32)

    func benchAsync(_ model: MLModel, _ input: MLFeatureProvider, n: Int = 200) async throws -> [Double] {
        for _ in 0..<10 { let _ = try await model.prediction(from: input) }
        var times = [Double]()
        for _ in 0..<n {
            let t0 = CFAbsoluteTimeGetCurrent()
            let _ = try await model.prediction(from: input)
            times.append(CFAbsoluteTimeGetCurrent() - t0)
        }
        return times
    }

    func report(_ label: String, _ times: [Double]) {
        let m = median(times) * 1000
        print("  \(label.padding(toLength: 46, withPad: " ", startingAt: 0))" +
              "median \(String(format: "%6.2f", m))ms")
    }

    // Load models
    let yoloALL = try await loadModel("yolo11s-seg.mlpackage", .all)
    let poseALL = try await loadModel("rtmpose.mlpackage", .all)

    let yoloIn = yoloALL.modelDescription.inputDescriptionsByName.keys.first!
    let yoloFP = try MLDictionaryFeatureProvider(dictionary: [
        yoloIn: MLFeatureValue(pixelBuffer: pb)
    ])
    let poseFP = try MLDictionaryFeatureProvider(dictionary: [
        "input_1": MLFeatureValue(multiArray: poseArr)
    ])

    // Preallocate output backings for RTMPose
    let poseOutNames = Array(poseALL.modelDescription.outputDescriptionsByName.keys).sorted()
    let outX = try MLMultiArray(shape: [1, 133, 576], dataType: .float32)
    let outY = try MLMultiArray(shape: [1, 133, 768], dataType: .float32)
    let poseOpts = MLPredictionOptions()
    poseOpts.outputBackings = [poseOutNames[0]: outX, poseOutNames[1]: outY]

    // ==========================================
    print("=== Standalone async (200 iters) ===\n")
    report("YOLO (ALL)",                    try await benchAsync(yoloALL, yoloFP))
    report("RTMPose (ALL)",                 try await benchAsync(poseALL, poseFP))

    // ==========================================
    print("\n=== Alternating async: YOLO → RTMPose (200 iters) ===\n")
    for _ in 0..<10 {
        let _ = try await yoloALL.prediction(from: yoloFP)
        let _ = try await poseALL.prediction(from: poseFP)
    }
    var aYT = [Double](), aPT = [Double]()
    for _ in 0..<200 {
        let t0 = CFAbsoluteTimeGetCurrent()
        let _ = try await yoloALL.prediction(from: yoloFP)
        let t1 = CFAbsoluteTimeGetCurrent()
        let _ = try await poseALL.prediction(from: poseFP)
        let t2 = CFAbsoluteTimeGetCurrent()
        aYT.append(t1 - t0); aPT.append(t2 - t1)
    }
    let aym = median(aYT) * 1000, apm = median(aPT) * 1000
    print("  YOLO:    \(String(format: "%6.2f", aym))ms")
    print("  RTMPose: \(String(format: "%6.2f", apm))ms")
    print("  Total:   \(String(format: "%6.2f", aym + apm))ms")

    // ==========================================
    print("\n=== Alternating async + outputBackings (200 iters) ===\n")
    for _ in 0..<10 {
        let _ = try await yoloALL.prediction(from: yoloFP)
        let _ = try await poseALL.prediction(from: poseFP, options: poseOpts)
    }
    var abYT = [Double](), abPT = [Double]()
    for _ in 0..<200 {
        let t0 = CFAbsoluteTimeGetCurrent()
        let _ = try await yoloALL.prediction(from: yoloFP)
        let t1 = CFAbsoluteTimeGetCurrent()
        let _ = try await poseALL.prediction(from: poseFP, options: poseOpts)
        let t2 = CFAbsoluteTimeGetCurrent()
        abYT.append(t1 - t0); abPT.append(t2 - t1)
    }
    let abym = median(abYT) * 1000, abpm = median(abPT) * 1000
    print("  YOLO:    \(String(format: "%6.2f", abym))ms")
    print("  RTMPose: \(String(format: "%6.2f", abpm))ms")
    print("  Total:   \(String(format: "%6.2f", abym + abpm))ms")

    // ==========================================
    let ym = median(try await benchAsync(yoloALL, yoloFP)) * 1000
    let pm = median(try await benchAsync(poseALL, poseFP)) * 1000
    print("\n=== Summary ===\n")
    print("  Standalone sum:             \(String(format: "%.1f", ym + pm))ms")
    print("  Alternating async:          \(String(format: "%.1f", aym + apm))ms")
    print("  Alternating async+backings: \(String(format: "%.1f", abym + abpm))ms")
    print("  Previous sync alternating:  ~35-37ms (for reference)")
}
