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

    func alternatingBench(_ yolo: MLModel, _ yoloIn: MLFeatureProvider,
                          _ pose: MLModel, _ poseIn: MLFeatureProvider,
                          label: String) async throws -> (Double, Double) {
        for _ in 0..<10 {
            let _ = try await yolo.prediction(from: yoloIn)
            let _ = try await pose.prediction(from: poseIn)
        }
        var yT = [Double](), pT = [Double]()
        for _ in 0..<200 {
            let t0 = CFAbsoluteTimeGetCurrent()
            let _ = try await yolo.prediction(from: yoloIn)
            let t1 = CFAbsoluteTimeGetCurrent()
            let _ = try await pose.prediction(from: poseIn)
            let t2 = CFAbsoluteTimeGetCurrent()
            yT.append(t1 - t0); pT.append(t2 - t1)
        }
        let ym = median(yT) * 1000, pm = median(pT) * 1000
        print("\n--- \(label) ---")
        print("  YOLO:    \(String(format: "%6.2f", ym))ms")
        print("  RTMPose: \(String(format: "%6.2f", pm))ms")
        print("  Total:   \(String(format: "%6.2f", ym + pm))ms")
        return (ym, pm)
    }

    // Load all model variants
    let yoloALL  = try await loadModel("yolo11s-seg.mlpackage", .all)
    let poseALL  = try await loadModel("rtmpose.mlpackage", .all)
    let poseFP16 = try await loadModel("rtmpose_fp16.mlpackage", .all)
    let poseCPU  = try await loadModel("rtmpose.mlpackage", .cpuOnly)
    let poseCNE  = try await loadModel("rtmpose.mlpackage", .cpuAndNeuralEngine)
    let fp16CNE  = try await loadModel("rtmpose_fp16.mlpackage", .cpuAndNeuralEngine)

    let yoloIn = yoloALL.modelDescription.inputDescriptionsByName.keys.first!
    let yoloFP = try MLDictionaryFeatureProvider(dictionary: [
        yoloIn: MLFeatureValue(pixelBuffer: pb)
    ])
    let poseFP = try MLDictionaryFeatureProvider(dictionary: [
        "input_1": MLFeatureValue(multiArray: poseArr)
    ])

    // ==========================================
    print("=== Standalone async (200 iters) ===\n")
    report("YOLO (ALL)",                    try await benchAsync(yoloALL, yoloFP))
    report("RTMPose fp32 (ALL)",            try await benchAsync(poseALL, poseFP))
    report("RTMPose fp16 (ALL)",            try await benchAsync(poseFP16, poseFP))
    report("RTMPose fp32 (CPU only)",       try await benchAsync(poseCPU, poseFP))
    report("RTMPose fp32 (CPU+NE)",         try await benchAsync(poseCNE, poseFP))
    report("RTMPose fp16 (CPU+NE)",         try await benchAsync(fp16CNE, poseFP))

    // ==========================================
    print("\n=== Alternating benchmarks ===")

    let (_, baseP) = try await alternatingBench(
        yoloALL, yoloFP, poseALL, poseFP,
        label: "YOLO(ALL) → Pose fp32(ALL) [baseline]")

    try await alternatingBench(
        yoloALL, yoloFP, poseFP16, poseFP,
        label: "YOLO(ALL) → Pose fp16(ALL)")

    try await alternatingBench(
        yoloALL, yoloFP, fp16CNE, poseFP,
        label: "YOLO(ALL) → Pose fp16(CPU+NE)")

    try await alternatingBench(
        yoloALL, yoloFP, poseCPU, poseFP,
        label: "YOLO(ALL) → Pose fp32(CPU only)")

    try await alternatingBench(
        yoloALL, yoloFP, poseCNE, poseFP,
        label: "YOLO(ALL) → Pose fp32(CPU+NE)")
}
