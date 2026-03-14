import Foundation
import CoreML
import CoreVideo
import Accelerate

// MARK: - YOLO Detection

public struct Detection {
    public let bbox: (Float, Float, Float, Float) // x1,y1,x2,y2 image coords
    public let confidence: Float
    public let coeffs: [Float]                    // 32 mask coefficients
    public let proto: MLMultiArray               // (1,32,160,160) prototype tensor
}

public func parseDetections(_ prediction: MLFeatureProvider,
                             outputNames: [String],
                             imgW: Int, imgH: Int,
                             lbScale: Float, padLeft: Int, padTop: Int) -> Detection? {
    var detArr: MLMultiArray?
    var protoArr: MLMultiArray?
    for name in outputNames {
        guard let a = prediction.featureValue(for: name)?.multiArrayValue else { continue }
        if a.shape.count == 3 { detArr = a }
        else if a.shape.count == 4 { protoArr = a }
    }
    guard let det = detArr, let proto = protoArr else { return nil }

    let nAnchors = det.shape[2].intValue
    let ptr = det.dataPointer.bindMemory(to: Float.self, capacity: det.count)

    let personOff = (4 + PERSON_CLASS) * nAnchors
    var bestIdx = -1
    var bestScore = YOLO_CONF_THRESH
    for i in 0..<nAnchors {
        let s = ptr[personOff + i]
        if s > bestScore { bestScore = s; bestIdx = i }
    }
    guard bestIdx >= 0 else { return nil }

    let cx = ptr[0 * nAnchors + bestIdx]
    let cy = ptr[1 * nAnchors + bestIdx]
    let w  = ptr[2 * nAnchors + bestIdx]
    let h  = ptr[3 * nAnchors + bestIdx]
    let x1 = max(0, (cx - w/2 - Float(padLeft)) / lbScale)
    let y1 = max(0, (cy - h/2 - Float(padTop))  / lbScale)
    let x2 = min(Float(imgW), (cx + w/2 - Float(padLeft)) / lbScale)
    let y2 = min(Float(imgH), (cy + h/2 - Float(padTop))  / lbScale)

    var coeffs = [Float](repeating: 0, count: 32)
    for c in 0..<32 { coeffs[c] = ptr[(84 + c) * nAnchors + bestIdx] }

    return Detection(bbox: (x1, y1, x2, y2), confidence: bestScore,
                     coeffs: coeffs, proto: proto)
}

// MARK: - Raw Mask

public struct MaskData {
    public let values: [Float]  // protoCW * protoCH, sigmoid values [0..1]
    public let width: Int       // protoCW
    public let height: Int      // protoCH
}

public func decodeMask(_ det: Detection,
                        imgW: Int, imgH: Int,
                        lbScale: Float, padLeft: Int, padTop: Int) -> MaskData {
    let protoScale = Float(160) / Float(YOLO_SIZE)
    let protoPadL = Int(Float(padLeft) * protoScale)
    let protoPadT = Int(Float(padTop) * protoScale)
    let protoCW   = Int(Float(Int(Float(imgW) * lbScale)) * protoScale)
    let protoCH   = Int(Float(Int(Float(imgH) * lbScale)) * protoScale)

    let proto = det.proto
    let pPtr = proto.dataPointer.bindMemory(to: Float.self, capacity: proto.count)
    let maskSize = 160 * 160

    var mask160 = [Float](repeating: 0, count: maskSize)
    for c in 0..<32 {
        var coeff = det.coeffs[c]
        vDSP_vsma(pPtr.advanced(by: c * maskSize), 1,
                  &coeff, mask160, 1, &mask160, 1, vDSP_Length(maskSize))
    }
    for i in 0..<maskSize { mask160[i] = 1.0 / (1.0 + exp(-mask160[i])) }

    var cropped = [Float](repeating: 0, count: protoCW * protoCH)
    for y in 0..<protoCH {
        for x in 0..<protoCW {
            cropped[y * protoCW + x] = mask160[(y + protoPadT) * 160 + (x + protoPadL)]
        }
    }
    return MaskData(values: cropped, width: protoCW, height: protoCH)
}

// MARK: - Contour Extraction

public func decodeContour(_ det: Detection,
                           imgW: Int, imgH: Int,
                           lbScale: Float, padLeft: Int, padTop: Int) -> [(Float, Float)] {
    let protoScale = Float(160) / Float(YOLO_SIZE)
    let protoPadL = Int(Float(padLeft) * protoScale)
    let protoPadT = Int(Float(padTop) * protoScale)
    let protoCW   = Int(Float(Int(Float(imgW) * lbScale)) * protoScale)
    let protoCH   = Int(Float(Int(Float(imgH) * lbScale)) * protoScale)
    let protoToImgX = Float(imgW) / Float(protoCW)
    let protoToImgY = Float(imgH) / Float(protoCH)

    let proto = det.proto
    let pPtr = proto.dataPointer.bindMemory(to: Float.self, capacity: proto.count)
    let maskSize = 160 * 160

    var mask = [Float](repeating: 0, count: maskSize)
    for c in 0..<32 {
        var coeff = det.coeffs[c]
        vDSP_vsma(pPtr.advanced(by: c * maskSize), 1,
                  &coeff, mask, 1, &mask, 1, vDSP_Length(maskSize))
    }
    for i in 0..<maskSize { mask[i] = 1.0 / (1.0 + exp(-mask[i])) }

    var binary = [UInt8](repeating: 0, count: protoCW * protoCH)
    for y in 0..<protoCH {
        for x in 0..<protoCW {
            if mask[(y + protoPadT) * 160 + (x + protoPadL)] > 0.5 {
                binary[y * protoCW + x] = 1
            }
        }
    }

    let raw = traceContour(binary, protoCW, protoCH)
    return raw.map { (Float($0) * protoToImgX, Float($1) * protoToImgY) }
}

public func traceContour(_ mask: [UInt8], _ w: Int, _ h: Int) -> [(Int, Int)] {
    let dx = [1, 1, 0, -1, -1, -1, 0, 1]
    let dy = [0, 1, 1,  1,  0, -1, -1, -1]

    var sx = -1, sy = -1
    for y in 0..<h {
        for x in 0..<w {
            if mask[y * w + x] != 0 { sx = x; sy = y; break }
        }
        if sx >= 0 { break }
    }
    guard sx >= 0 else { return [] }

    var contour: [(Int, Int)] = [(sx, sy)]
    var x = sx, y = sy, back = 4
    for _ in 0..<(w * h * 2) {
        let start = (back + 1) % 8
        var moved = false
        for i in 0..<8 {
            let d = (start + i) % 8
            let nx = x + dx[d], ny = y + dy[d]
            if nx >= 0, nx < w, ny >= 0, ny < h, mask[ny * w + nx] != 0 {
                x = nx; y = ny; back = (d + 4) % 8
                if x == sx && y == sy { return contour }
                contour.append((x, y)); moved = true; break
            }
        }
        if !moved { break }
    }
    return contour
}
