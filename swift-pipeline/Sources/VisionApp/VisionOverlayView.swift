import SwiftUI
import Vision
import CoreVideo

struct VisionOverlayView: View {
    let result: VisionFrameResult?
    let viewSize: CGSize
    let imageSize: CGSize
    let mirrored: Bool

    func mapX(_ x: Double, _ scaleX: Double, _ width: Double) -> Double {
        let mapped = x * scaleX
        return mirrored ? (width - mapped) : mapped
    }

    // MARK: - Mask Rendering

    func maskToCGImage(_ pixelBuffer: CVPixelBuffer) -> CGImage? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let w = CVPixelBufferGetWidth(pixelBuffer)
        let h = CVPixelBufferGetHeight(pixelBuffer)
        let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer)?
                .assumingMemoryBound(to: UInt8.self) else { return nil }

        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        for y in 0..<h {
            for x in 0..<w {
                let alpha = base[y * rowBytes + x]
                let scaled = UInt8(min(255, Int(alpha) * 160 / 255))
                let i = (y * w + x) * 4
                rgba[i]     = 0       // R
                rgba[i + 1] = scaled  // G (premultiplied)
                rgba[i + 2] = 0       // B
                rgba[i + 3] = scaled  // A
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        return CGImage(width: w, height: h,
                       bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: w * 4,
                       space: colorSpace,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                       provider: provider,
                       decode: nil, shouldInterpolate: true,
                       intent: .defaultIntent)
    }

    // MARK: - Body

    var body: some View {
        Canvas { context, size in
            guard let r = result else { return }

            let scaleX = size.width / imageSize.width
            let scaleY = size.height / imageSize.height

            // 1. Draw mask (bottom layer)
            if let maskPB = r.maskPixelBuffer, let cgImage = maskToCGImage(maskPB) {
                let img = Image(decorative: cgImage, scale: 1)
                if mirrored {
                    var ctx = context
                    ctx.concatenate(CGAffineTransform(scaleX: -1, y: 1)
                        .translatedBy(x: -size.width, y: 0))
                    ctx.draw(img, in: CGRect(origin: .zero, size: size))
                } else {
                    context.draw(img, in: CGRect(origin: .zero, size: size))
                }
            }

            // 1.5. Draw contour outlines
            if let cgPath = r.contourPath {
                let transform: CGAffineTransform
                if mirrored {
                    transform = CGAffineTransform(scaleX: -size.width, y: -size.height)
                        .concatenating(CGAffineTransform(translationX: size.width, y: size.height))
                } else {
                    transform = CGAffineTransform(scaleX: size.width, y: -size.height)
                        .concatenating(CGAffineTransform(translationX: 0, y: size.height))
                }
                var t = transform
                if let transformed = cgPath.copy(using: &t) {
                    let path = Path(transformed)
                    context.stroke(path, with: .color(.white), lineWidth: 2)
                }
            }

            // 2. Draw body skeletons
            let bodyColors: [Color] = [.red, .yellow, .orange, .orange, .blue, .blue]
            for bodyPose in r.bodyPoses {
                // Draw bones
                for (j1, j2) in BodySkeleton.connections {
                    guard let (p1, c1) = bodyPose.joints[j1.rawValue.rawValue],
                          let (p2, c2) = bodyPose.joints[j2.rawValue.rawValue],
                          c1 > 0.3, c1 <= 1.0, c2 > 0.3, c2 <= 1.0 else { continue }
                    let from = CGPoint(x: mapX(p1.x, scaleX, size.width), y: p1.y * scaleY)
                    let to = CGPoint(x: mapX(p2.x, scaleX, size.width), y: p2.y * scaleY)
                    var path = Path()
                    path.move(to: from)
                    path.addLine(to: to)
                    context.stroke(path, with: .color(.white.opacity(0.7)), lineWidth: 2)
                }

                // Draw joints
                for (jointName, (pt, conf)) in bodyPose.joints {
                    if conf < 0.3 || conf > 1.0 { continue }
                    let x = mapX(pt.x, scaleX, size.width)
                    let y = pt.y * scaleY

                    // Determine color from joint name
                    let jn = VNHumanBodyPoseObservation.JointName(rawValue:
                        VNHumanBodyPoseObservation.JointName.RawValue(rawValue: jointName))
                    let group = BodySkeleton.colorGroup(jn)
                    let color = bodyColors[group]

                    let circle = Path(ellipseIn: CGRect(x: x - 4, y: y - 4, width: 8, height: 8))
                    context.fill(circle, with: .color(color))
                }
            }

            // 3. Draw hand skeletons
            for handPose in r.handPoses {
                let handColor: Color = handPose.chirality == .left ? .cyan : .mint

                // Draw bones. Upper-bound filter intentionally omitted: a Vision bug
                // sometimes returns uninitialized confidences (values like 191-255)
                // for joints whose positions are still valid; only filter true zeros.
                for (j1, j2) in HandSkeleton.connections {
                    guard let (p1, c1) = handPose.joints[j1.rawValue.rawValue],
                          let (p2, c2) = handPose.joints[j2.rawValue.rawValue],
                          c1 > 0.3, c2 > 0.3 else { continue }
                    let from = CGPoint(x: mapX(p1.x, scaleX, size.width), y: p1.y * scaleY)
                    let to = CGPoint(x: mapX(p2.x, scaleX, size.width), y: p2.y * scaleY)
                    var path = Path()
                    path.move(to: from)
                    path.addLine(to: to)
                    context.stroke(path, with: .color(handColor.opacity(0.7)), lineWidth: 1.5)
                }

                // Draw joints
                for (_, (pt, conf)) in handPose.joints {
                    if conf < 0.3 { continue }
                    let x = mapX(pt.x, scaleX, size.width)
                    let y = pt.y * scaleY
                    let circle = Path(ellipseIn: CGRect(x: x - 2.5, y: y - 2.5, width: 5, height: 5))
                    context.fill(circle, with: .color(handColor))
                }
            }

            // 4. Draw face landmarks
            for face in r.faceLandmarks {
                for pt in face.allPoints {
                    let x = mapX(pt.x, scaleX, size.width)
                    let y = pt.y * scaleY
                    let dot = Path(ellipseIn: CGRect(x: x - 1.5, y: y - 1.5, width: 3, height: 3))
                    context.fill(dot, with: .color(.purple))
                }
            }
        }
    }
}
