import SwiftUI
import Vision

struct VisionCameraView: View {
    @StateObject private var engine = VisionPipelineEngine()

    let qualityLabels = ["Fast", "Balanced", "Accurate"]

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
            .allowsHitTesting(false)

            // Info + controls overlay
            VStack {
                HStack(alignment: .top) {
                    // Left: controls
                    VStack(alignment: .leading, spacing: 6) {
                        // Camera picker
                        Picker("Camera", selection: $engine.selectedCameraId) {
                            ForEach(engine.availableCameras) { cam in
                                Text(cam.name).tag(cam.id)
                            }
                        }
                        .frame(width: 220)

                        Spacer().frame(height: 2)

                        Toggle("Mask", isOn: $engine.enableSeg)
                        if engine.enableSeg {
                            HStack(spacing: 4) {
                                ForEach(0..<3) { i in
                                    Button(qualityLabels[i]) {
                                        engine.segQualityIndex = i
                                        engine.syncConfig()
                                    }
                                    .buttonStyle(.bordered)
                                    .tint(engine.segQualityIndex == i ? .green : .gray)
                                }
                            }
                        }
                        Toggle("Body", isOn: $engine.enableBody)
                        Toggle("Hands", isOn: $engine.enableHands)
                        Toggle("Face", isOn: $engine.enableFace)

                        Spacer().frame(height: 2)

                        Toggle("Batch mode", isOn: $engine.batchMode)

                        Spacer().frame(height: 2)

                        Toggle("Syphon out", isOn: $engine.enableSyphon)
                        if engine.enableSyphon {
                            HStack {
                                Text(String(format: "Threshold: %.0f%%", engine.maskThreshold * 100))
                                Slider(value: $engine.maskThreshold, in: 0...1)
                                    .frame(width: 120)
                            }
                        }
                    }
                    .font(.system(.caption, design: .monospaced))
                    .foregroundColor(.white)
                    .padding(8)

                    Spacer()

                    // Right: timing
                    VStack(alignment: .trailing, spacing: 3) {
                        Text(String(format: "%5.1f FPS", engine.fps))
                            .font(.system(.title2, design: .monospaced))
                            .foregroundColor(.white)
                        if let r = engine.latestResult {
                            Text(String(format: "total: %6.1f ms", r.frameTimeMs))
                                .foregroundColor(.white)
                        }
                        if engine.batchMode {
                            Text("  (batched)")
                                .foregroundColor(.orange)
                        } else {
                            HStack(spacing: 12) {
                                timingLabel("seg ", engine.segMs, engine.enableSeg || engine.enableSyphon)
                                timingLabel("body", engine.bodyMs, engine.enableBody)
                            }
                            HStack(spacing: 12) {
                                timingLabel("hand", engine.handMs, engine.enableHands)
                                timingLabel("face", engine.faceMs, engine.enableFace)
                            }
                        }
                        if let r = engine.latestResult {
                            let bodyCount = r.bodyPoses.first?.joints.count ?? 0
                            let handCount = r.handPoses.reduce(0) { $0 + $1.joints.count }
                            let faceCount = r.faceLandmarks.reduce(0) { $0 + $1.allPoints.count }
                            Text(String(format: "%2db %2dh %2df pts", bodyCount, handCount, faceCount))
                                .foregroundColor(.gray)
                        }
                        if engine.enableSyphon {
                            Text("syphon: \"VisionApp\"")
                                .foregroundColor(.orange)
                        }
                    }
                    .font(.system(.caption, design: .monospaced))
                    .padding(8)
                }
                Spacer()
            }
            .padding()
        }
        .onChange(of: engine.enableSeg) { _ in engine.syncConfig() }
        .onChange(of: engine.enableBody) { _ in engine.syncConfig() }
        .onChange(of: engine.enableHands) { _ in engine.syncConfig() }
        .onChange(of: engine.enableFace) { _ in engine.syncConfig() }
        .onChange(of: engine.batchMode) { _ in engine.syncConfig() }
        .onChange(of: engine.enableSyphon) { _ in engine.syncConfig() }
        .onChange(of: engine.maskThreshold) { _ in engine.syncConfig() }
        .onChange(of: engine.selectedCameraId) { newId in engine.switchCamera(to: newId) }
        .task {
            do {
                try engine.start()
            } catch {
                print("Failed to start: \(error)")
            }
        }
    }

    func timingLabel(_ name: String, _ ms: Double, _ enabled: Bool) -> some View {
        Text(enabled ? String(format: "%@:%6.1f ms", name, ms) : "\(name):   off")
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
