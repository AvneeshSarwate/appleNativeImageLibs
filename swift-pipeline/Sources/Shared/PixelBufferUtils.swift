import Foundation
import CoreVideo
import Accelerate

public func makePixelBuffer(_ w: Int, _ h: Int) -> CVPixelBuffer {
    var pb: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, w, h,
                        kCVPixelFormatType_32BGRA,
                        [kCVPixelBufferIOSurfacePropertiesKey: [:] as [String: Any]] as CFDictionary,
                        &pb)
    return pb!
}

/// Letterbox a BGRA CVPixelBuffer into a 640x640 CVPixelBuffer using vImage.
public func letterboxPixelBuffer(src: CVPixelBuffer, into dest: CVPixelBuffer,
                                  imgW: Int, imgH: Int) -> (Float, Int, Int) {
    let scale = min(Float(YOLO_SIZE) / Float(imgH), Float(YOLO_SIZE) / Float(imgW))
    let newW = Int(Float(imgW) * scale)
    let newH = Int(Float(imgH) * scale)
    let padLeft = (YOLO_SIZE - newW) / 2
    let padTop  = (YOLO_SIZE - newH) / 2

    CVPixelBufferLockBaseAddress(src, .readOnly)
    CVPixelBufferLockBaseAddress(dest, [])

    let srcBase = CVPixelBufferGetBaseAddress(src)!
    let srcRowBytes = CVPixelBufferGetBytesPerRow(src)
    let destBase = CVPixelBufferGetBaseAddress(dest)!
    let destRowBytes = CVPixelBufferGetBytesPerRow(dest)

    // Fill with grey
    var pattern: UInt32 = 0
    withUnsafeMutableBytes(of: &pattern) { p in
        p[0] = 114; p[1] = 114; p[2] = 114; p[3] = 255
    }
    memset_pattern4(destBase, &pattern, destRowBytes * YOLO_SIZE)

    // Resize source into content region
    var srcBuf = vImage_Buffer(data: srcBase, height: vImagePixelCount(imgH),
                               width: vImagePixelCount(imgW), rowBytes: srcRowBytes)
    var dstBuf = vImage_Buffer(
        data: destBase.advanced(by: padTop * destRowBytes + padLeft * 4),
        height: vImagePixelCount(newH), width: vImagePixelCount(newW), rowBytes: destRowBytes)
    vImageScale_ARGB8888(&srcBuf, &dstBuf, nil, vImage_Flags(kvImageNoFlags))

    CVPixelBufferUnlockBaseAddress(src, .readOnly)
    CVPixelBufferUnlockBaseAddress(dest, [])

    return (scale, padLeft, padTop)
}

/// Letterbox raw BGRA Data into a 640x640 CVPixelBuffer using vImage.
public func letterbox(frameData: Data, into dest: CVPixelBuffer,
                       imgW: Int, imgH: Int) -> (Float, Int, Int) {
    let scale = min(Float(YOLO_SIZE) / Float(imgH), Float(YOLO_SIZE) / Float(imgW))
    let newW = Int(Float(imgW) * scale)
    let newH = Int(Float(imgH) * scale)
    let padLeft = (YOLO_SIZE - newW) / 2
    let padTop  = (YOLO_SIZE - newH) / 2

    CVPixelBufferLockBaseAddress(dest, [])
    let base = CVPixelBufferGetBaseAddress(dest)!
    let rowBytes = CVPixelBufferGetBytesPerRow(dest)

    var pattern: UInt32 = 0
    withUnsafeMutableBytes(of: &pattern) { p in
        p[0] = 114; p[1] = 114; p[2] = 114; p[3] = 255
    }
    memset_pattern4(base, &pattern, rowBytes * YOLO_SIZE)

    frameData.withUnsafeBytes { src in
        var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src.baseAddress!),
                                   height: vImagePixelCount(imgH), width: vImagePixelCount(imgW),
                                   rowBytes: imgW * 4)
        var dstBuf = vImage_Buffer(
            data: base.advanced(by: padTop * rowBytes + padLeft * 4),
            height: vImagePixelCount(newH), width: vImagePixelCount(newW), rowBytes: rowBytes)
        vImageScale_ARGB8888(&srcBuf, &dstBuf, nil, vImage_Flags(kvImageNoFlags))
    }

    CVPixelBufferUnlockBaseAddress(dest, [])
    return (scale, padLeft, padTop)
}
