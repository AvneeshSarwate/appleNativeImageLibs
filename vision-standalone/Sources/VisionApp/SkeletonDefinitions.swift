import Vision

// MARK: - Body Skeleton

struct BodySkeleton {
    static let connections: [(VNHumanBodyPoseObservation.JointName, VNHumanBodyPoseObservation.JointName)] = [
        // Head
        (.leftEar, .leftEye),    (.rightEar, .rightEye),
        (.leftEye, .nose),       (.rightEye, .nose),
        // Spine
        (.nose, .neck),
        (.neck, .root),
        // Shoulders
        (.neck, .leftShoulder),  (.neck, .rightShoulder),
        // Left arm
        (.leftShoulder, .leftElbow), (.leftElbow, .leftWrist),
        // Right arm
        (.rightShoulder, .rightElbow), (.rightElbow, .rightWrist),
        // Left leg
        (.root, .leftHip), (.leftHip, .leftKnee), (.leftKnee, .leftAnkle),
        // Right leg
        (.root, .rightHip), (.rightHip, .rightKnee), (.rightKnee, .rightAnkle),
    ]

    /// Map joint names to color groups for rendering
    static func colorGroup(_ joint: VNHumanBodyPoseObservation.JointName) -> Int {
        switch joint {
        case .nose, .leftEye, .rightEye, .leftEar, .rightEar: return 0  // head
        case .neck, .root: return 1  // torso
        case .leftShoulder, .leftElbow, .leftWrist: return 2  // left arm
        case .rightShoulder, .rightElbow, .rightWrist: return 3  // right arm
        case .leftHip, .leftKnee, .leftAnkle: return 4  // left leg
        case .rightHip, .rightKnee, .rightAnkle: return 5  // right leg
        default: return 1
        }
    }
}

// MARK: - Hand Skeleton

struct HandSkeleton {
    static let connections: [(VNHumanHandPoseObservation.JointName, VNHumanHandPoseObservation.JointName)] = [
        // Thumb
        (.wrist, .thumbCMC), (.thumbCMC, .thumbMP),
        (.thumbMP, .thumbIP), (.thumbIP, .thumbTip),
        // Index
        (.wrist, .indexMCP), (.indexMCP, .indexPIP),
        (.indexPIP, .indexDIP), (.indexDIP, .indexTip),
        // Middle
        (.wrist, .middleMCP), (.middleMCP, .middlePIP),
        (.middlePIP, .middleDIP), (.middleDIP, .middleTip),
        // Ring
        (.wrist, .ringMCP), (.ringMCP, .ringPIP),
        (.ringPIP, .ringDIP), (.ringDIP, .ringTip),
        // Little
        (.wrist, .littleMCP), (.littleMCP, .littlePIP),
        (.littlePIP, .littleDIP), (.littleDIP, .littleTip),
    ]
}
