import Foundation

// Input dimensions
public let YOLO_SIZE = 640
public let YOLO_CONF_THRESH: Float = 0.25
public let PERSON_CLASS = 0
public let SIMCC_SPLIT: Float = 2.0

// RTMPose normalization (BGR order)
public let POSE_MEAN: [Float] = [123.675, 116.28, 103.53]
public let POSE_STD: [Float]  = [58.395,  57.12,  57.375]
