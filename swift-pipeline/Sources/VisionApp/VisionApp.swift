import SwiftUI
import Vision

struct VisionCameraView: View {
    @StateObject private var engine = VisionPipelineEngine()

    var body: some View {
        ZStack {
            Color.black.edgesIgnoringSafeArea(.all)

            VisionOverlayView(
                result: engine.latestResult,
                viewSize: CGSize(width: 960, height: 540),
                imageSize: engine.imageSize,
                mirrored: true
            )
            .frame(width: 960, height: 540)

            // Info + controls overlay
            VStack {
                HStack(alignment: .top) {
                    // Left: toggles
                    VStack(alignment: .leading, spacing: 6) {
                        Toggle("Mask", isOn: $engine.enableSeg)
                        if engine.enableSeg {
                            Picker("Quality", selection: $engine.segQuality) {
                                Text("Fast").tag(VNGeneratePersonSegmentationRequest.QualityLevel.fast)
                                Text("Balanced").tag(VNGeneratePersonSegmentationRequest.QualityLevel.balanced)
                                Text("Accurate").tag(VNGeneratePersonSegmentationRequest.QualityLevel.accurate)
                            }
                            .pickerStyle(.segmented)
                            .frame(width: 200)
                        }
                        Toggle("Body", isOn: $engine.enableBody)
                        Toggle("Hands", isOn: $engine.enableHands)
                        Toggle("Face", isOn: $engine.enableFace)
                    }
                    .font(.system(.caption, design: .monospaced))
                    .foregroundColor(.white)
                    .padding(8)
                    .background(Color.black.opacity(0.6))
                    .cornerRadius(8)

                    Spacer()

                    // Right: timing
                    VStack(alignment: .trailing, spacing: 3) {
                        Text(String(format: "%.1f FPS", engine.fps))
                            .font(.system(.title2, design: .monospaced))
                            .foregroundColor(.white)
                        if let r = engine.latestResult {
                            Text(String(format: "total: %.1f ms", r.frameTimeMs))
                                .foregroundColor(.white)
                        }
                        HStack(spacing: 12) {
                            timingLabel("seg", engine.segMs, engine.enableSeg)
                            timingLabel("body", engine.bodyMs, engine.enableBody)
                        }
                        HStack(spacing: 12) {
                            timingLabel("hand", engine.handMs, engine.enableHands)
                            timingLabel("face", engine.faceMs, engine.enableFace)
                        }
                        if let r = engine.latestResult {
                            let bodyCount = r.bodyPoses.first?.joints.count ?? 0
                            let handCount = r.handPoses.reduce(0) { $0 + $1.joints.count }
                            let faceCount = r.faceLandmarks.reduce(0) { $0 + $1.allPoints.count }
                            Text("\(bodyCount)b \(handCount)h \(faceCount)f pts")
                                .foregroundColor(.gray)
                        }
                    }
                    .font(.system(.caption, design: .monospaced))
                    .padding(8)
                    .background(Color.black.opacity(0.6))
                    .cornerRadius(8)
                }
                Spacer()
            }
            .padding()
        }
        .task {
            do {
                try engine.start()
            } catch {
                print("Failed to start: \(error)")
            }
        }
    }

    func timingLabel(_ name: String, _ ms: Double, _ enabled: Bool) -> some View {
        Text(enabled ? String(format: "%@: %.1f ms", name, ms) : "\(name): off")
            .foregroundColor(enabled ? .green : .gray)
    }
}

@main
struct VisionTestApp: App {
    var body: some Scene {
        WindowGroup {
            VisionCameraView()
                .frame(width: 960, height: 540)
        }
    }
}
