import Foundation
import CoreML
import Accelerate

// MARK: - Pose Result

public struct PoseResult {
    public let keypoints: [(Float, Float)]  // 133 points in image coords
    public let confidence: [Float]
}

// MARK: - Pose Inference

/// Run pose estimation on a BGRA CVPixelBuffer, cropping to bbox.
public func runPoseOnPixelBuffer(
    model: MLModel, inputName: String, outputNames: [String],
    pixelBuffer: CVPixelBuffer, bbox: (Float, Float, Float, Float),
    poseH: Int, poseW: Int
) async throws -> PoseResult? {
    let imgW = CVPixelBufferGetWidth(pixelBuffer)
    let imgH = CVPixelBufferGetHeight(pixelBuffer)

    let (bx1, by1, bx2, by2) = bbox
    let cx = (bx1 + bx2) / 2, cy = (by1 + by2) / 2
    let bw = bx2 - bx1, bh = by2 - by1
    let cropScale = max(bw / Float(poseW), bh / Float(poseH)) * 1.25
    let nw = Float(poseW) * cropScale, nh = Float(poseH) * cropScale

    let x1c = max(0, Int(cx - nw / 2))
    let y1c = max(0, Int(cy - nh / 2))
    let x2c = min(imgW, Int(cx + nw / 2))
    let y2c = min(imgH, Int(cy + nh / 2))
    let cropW = x2c - x1c, cropH = y2c - y1c
    guard cropW > 0, cropH > 0 else { return nil }

    // Crop + resize using vImage from CVPixelBuffer
    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    let srcBase = CVPixelBufferGetBaseAddress(pixelBuffer)!
    let srcRowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)

    var resized = [UInt8](repeating: 0, count: poseW * poseH * 4)
    resized.withUnsafeMutableBytes { dst in
        var cropBuf = vImage_Buffer(
            data: srcBase.advanced(by: y1c * srcRowBytes + x1c * 4),
            height: vImagePixelCount(cropH), width: vImagePixelCount(cropW),
            rowBytes: srcRowBytes)
        var dstBuf = vImage_Buffer(
            data: dst.baseAddress!, height: vImagePixelCount(poseH),
            width: vImagePixelCount(poseW), rowBytes: poseW * 4)
        vImageScale_ARGB8888(&cropBuf, &dstBuf, nil, vImage_Flags(kvImageNoFlags))
    }
    CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)

    // Normalize BGRA → CHW float32
    let pixels = poseW * poseH
    let arr = try MLMultiArray(shape: [1, 3, NSNumber(value: poseH), NSNumber(value: poseW)],
                               dataType: .float32)
    let fp = arr.dataPointer.bindMemory(to: Float.self, capacity: 3 * pixels)
    for i in 0..<pixels {
        fp[i]              = (Float(resized[i * 4])     - POSE_MEAN[0]) / POSE_STD[0]
        fp[pixels + i]     = (Float(resized[i * 4 + 1]) - POSE_MEAN[1]) / POSE_STD[1]
        fp[2 * pixels + i] = (Float(resized[i * 4 + 2]) - POSE_MEAN[2]) / POSE_STD[2]
    }

    let input = try MLDictionaryFeatureProvider(dictionary: [
        inputName: MLFeatureValue(multiArray: arr)
    ])
    let pred = try await model.prediction(from: input)

    let scaleX = Float(cropW) / Float(poseW)
    let scaleY = Float(cropH) / Float(poseH)
    return decodePose(pred, outputNames: outputNames,
                      scaleX: scaleX, scaleY: scaleY, x1c: x1c, y1c: y1c)
}

/// Run pose estimation from raw BGRA frame data, cropping to bbox.
public func runPoseOnFrameData(
    model: MLModel, inputName: String, outputNames: [String],
    frameData: Data, frameW: Int, frameH: Int,
    bbox: (Float, Float, Float, Float),
    poseH: Int, poseW: Int
) async throws -> PoseResult? {
    let (bx1, by1, bx2, by2) = bbox
    let cx = (bx1 + bx2) / 2, cy = (by1 + by2) / 2
    let bw = bx2 - bx1, bh = by2 - by1
    let cropScale = max(bw / Float(poseW), bh / Float(poseH)) * 1.25
    let nw = Float(poseW) * cropScale, nh = Float(poseH) * cropScale

    let x1c = max(0, Int(cx - nw / 2))
    let y1c = max(0, Int(cy - nh / 2))
    let x2c = min(frameW, Int(cx + nw / 2))
    let y2c = min(frameH, Int(cy + nh / 2))
    let cropW = x2c - x1c, cropH = y2c - y1c
    guard cropW > 0, cropH > 0 else { return nil }

    let resized: [UInt8] = frameData.withUnsafeBytes { src in
        var cropBuf = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: src.baseAddress!
                    .advanced(by: y1c * frameW * 4 + x1c * 4)),
            height: vImagePixelCount(cropH), width: vImagePixelCount(cropW),
            rowBytes: frameW * 4)
        var out = [UInt8](repeating: 0, count: poseW * poseH * 4)
        out.withUnsafeMutableBytes { dst in
            var dstBuf = vImage_Buffer(
                data: dst.baseAddress!, height: vImagePixelCount(poseH),
                width: vImagePixelCount(poseW), rowBytes: poseW * 4)
            vImageScale_ARGB8888(&cropBuf, &dstBuf, nil, vImage_Flags(kvImageNoFlags))
        }
        return out
    }

    let pixels = poseW * poseH
    let arr = try MLMultiArray(shape: [1, 3, NSNumber(value: poseH), NSNumber(value: poseW)],
                               dataType: .float32)
    let fp = arr.dataPointer.bindMemory(to: Float.self, capacity: 3 * pixels)
    for i in 0..<pixels {
        fp[i]              = (Float(resized[i * 4])     - POSE_MEAN[0]) / POSE_STD[0]
        fp[pixels + i]     = (Float(resized[i * 4 + 1]) - POSE_MEAN[1]) / POSE_STD[1]
        fp[2 * pixels + i] = (Float(resized[i * 4 + 2]) - POSE_MEAN[2]) / POSE_STD[2]
    }

    let input = try MLDictionaryFeatureProvider(dictionary: [
        inputName: MLFeatureValue(multiArray: arr)
    ])
    let pred = try await model.prediction(from: input)

    let scaleX = Float(cropW) / Float(poseW)
    let scaleY = Float(cropH) / Float(poseH)
    return decodePose(pred, outputNames: outputNames,
                      scaleX: scaleX, scaleY: scaleY, x1c: x1c, y1c: y1c)
}

// MARK: - SimCC Decode

func decodePose(_ pred: MLFeatureProvider, outputNames: [String],
                scaleX: Float, scaleY: Float, x1c: Int, y1c: Int) -> PoseResult? {
    guard let sxArr = pred.featureValue(for: outputNames[0])?.multiArrayValue,
          let syArr = pred.featureValue(for: outputNames[1])?.multiArrayValue
    else { return nil }

    let nKpts = sxArr.shape[1].intValue
    let xLen = sxArr.shape[2].intValue
    let yLen = syArr.shape[2].intValue
    let xp = sxArr.dataPointer.bindMemory(to: Float.self, capacity: sxArr.count)
    let yp = syArr.dataPointer.bindMemory(to: Float.self, capacity: syArr.count)

    var kpts = [(Float, Float)]()
    var conf = [Float]()
    kpts.reserveCapacity(nKpts)
    conf.reserveCapacity(nKpts)

    for k in 0..<nKpts {
        var maxXV: Float = 0, maxXI: vDSP_Length = 0
        var maxYV: Float = 0, maxYI: vDSP_Length = 0
        vDSP_maxvi(xp.advanced(by: k * xLen), 1, &maxXV, &maxXI, vDSP_Length(xLen))
        vDSP_maxvi(yp.advanced(by: k * yLen), 1, &maxYV, &maxYI, vDSP_Length(yLen))

        let px = Float(maxXI) / SIMCC_SPLIT * scaleX + Float(x1c)
        let py = Float(maxYI) / SIMCC_SPLIT * scaleY + Float(y1c)
        kpts.append((px, py))
        conf.append(min(maxXV, maxYV))
    }
    return PoseResult(keypoints: kpts, confidence: conf)
}
