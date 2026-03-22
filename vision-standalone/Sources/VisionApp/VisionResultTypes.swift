import Foundation
import CoreVideo

struct VisionFrameResult {
    let maskPixelBuffer: CVPixelBuffer?
    let bodyPoses: [BodyPoseData]
    let handPoses: [HandPoseData]
    let faceLandmarks: [FaceLandmarkData]
    let frameTimeMs: Double
}

struct BodyPoseData {
    /// jointName -> (pixel coords top-left origin, confidence)
    let joints: [String: (CGPoint, Float)]
}

struct HandPoseData {
    let joints: [String: (CGPoint, Float)]
    let chirality: HandChirality
}

enum HandChirality { case left, right, unknown }

struct FaceLandmarkData {
    /// All 76 face points in pixel coords (top-left origin)
    let allPoints: [CGPoint]
    /// Face bounding box in pixel coords
    let boundingBox: CGRect
}
