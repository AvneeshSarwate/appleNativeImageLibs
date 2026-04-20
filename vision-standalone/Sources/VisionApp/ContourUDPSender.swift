import Foundation
import Network
import Vision

/// Streams contour trees over WebSocket (binary frames) using a compact protocol.
///
/// Wire format (little-endian):
/// ```
/// HEADER (12 bytes)
///   [0..3]   UInt32  magic         0x434F4E54 ("CONT")
///   [4..7]   UInt32  frameNumber
///   [8..9]   UInt16  contourCount  N
///   [10..11] UInt16  reserved      0
///
/// CONTOUR DESCRIPTORS (N × 20 bytes)
///   [0..1]   UInt16  pointCount
///   [2..3]   Int16   parentIndex   (-1 = top-level)
///   [4..7]   UInt32  pointOffset   byte offset into point data section
///   [8..11]  Float32 centroidX     remapped + Y-flipped
///   [12..15] Float32 centroidY     remapped + Y-flipped
///   [16..19] Float32 area          signed area (from VNGeometryUtils)
///
/// POINT DATA (totalPoints × 8 bytes)
///   Packed Float32 pairs [x, y, x, y, ...]
/// ```
final class ContourWebSocketServer: @unchecked Sendable {
    private let listener: NWListener
    private let queue = DispatchQueue(label: "contour-ws-server")
    private var connections: [NWConnection] = []
    private let lock = NSLock()

    init(port: UInt16 = 9100) {
        let params = NWParameters.tcp
        let wsOptions = NWProtocolWebSocket.Options()
        params.defaultProtocolStack.applicationProtocols.insert(wsOptions, at: 0)

        listener = try! NWListener(using: params, on: NWEndpoint.Port(rawValue: port)!)
        listener.newConnectionHandler = { [weak self] conn in
            self?.handleNewConnection(conn)
        }
        listener.stateUpdateHandler = { state in
            switch state {
            case .ready:
                print("[ContourWS] server listening on port \(port)")
            case .failed(let err):
                print("[ContourWS] server failed: \(err)")
            default: break
            }
        }
        listener.start(queue: queue)
    }

    private func handleNewConnection(_ conn: NWConnection) {
        print("[ContourWS] client connected")
        conn.stateUpdateHandler = { [weak self] state in
            if case .failed(_) = state { self?.removeConnection(conn) }
            if case .cancelled = state { self?.removeConnection(conn) }
        }
        conn.start(queue: queue)
        lock.withLock { connections.append(conn) }
        receiveLoop(conn)
    }

    private func receiveLoop(_ conn: NWConnection) {
        conn.receiveMessage { [weak self] content, context, isComplete, error in
            if error != nil { self?.removeConnection(conn); return }
            self?.receiveLoop(conn)
        }
    }

    private func removeConnection(_ conn: NWConnection) {
        print("[ContourWS] client disconnected")
        conn.cancel()
        lock.withLock { connections.removeAll { $0 === conn } }
    }

    func send(obs: VNContoursObservation,
              padW: Int, padH: Int, mW: Int, mH: Int, border: Int,
              frameNumber: UInt32) {
        struct FlatContour {
            let parentIndex: Int16
            let contour: VNContour
        }

        var flat: [FlatContour] = []
        for top in obs.topLevelContours {
            guard top.pointCount >= 5 else { continue }
            let topIdx = Int16(flat.count)
            flat.append(FlatContour(parentIndex: -1, contour: top))
            for child in top.childContours {
                guard child.pointCount >= 5 else { continue }
                flat.append(FlatContour(parentIndex: topIdx, contour: child))
            }
        }

        guard !flat.isEmpty else { return }

        // Remap factors
        let scaleX = Float(padW) / Float(mW)
        let scaleY = Float(padH) / Float(mH)
        let offX = Float(border) / Float(mW)
        let offY = Float(border) / Float(mH)

        // Pre-compute per-contour metadata
        struct ContourMeta {
            let parentIndex: Int16
            let points: [simd_float2]
            let centroidX: Float
            let centroidY: Float
            let area: Float
        }

        let metas: [ContourMeta] = flat.map { fc in
            let pts = fc.contour.normalizedPoints

            // Compute centroid from raw normalized points, then remap
            var sx: Float = 0, sy: Float = 0
            for p in pts { sx += p.x; sy += p.y }
            let n = Float(pts.count)
            let cx = (sx / n) * scaleX - offX
            let cy = 1.0 - ((sy / n) * scaleY - offY)

            // Signed area via VNGeometryUtils (in normalized coords, pre-remap)
            var rawArea: Double = 0
            try? VNGeometryUtils.calculateArea(&rawArea, for: fc.contour, orientedArea: true)

            return ContourMeta(
                parentIndex: fc.parentIndex,
                points: pts,
                centroidX: cx,
                centroidY: cy,
                area: Float(rawArea)
            )
        }

        let totalPoints = metas.reduce(0) { $0 + $1.points.count }
        let headerSize = 12
        let descSize = metas.count * 20
        let pointDataSize = totalPoints * 8
        let totalSize = headerSize + descSize + pointDataSize

        var data = Data(count: totalSize)
        data.withUnsafeMutableBytes { raw in
            let ptr = raw.baseAddress!

            // Header
            ptr.storeBytes(of: UInt32(0x434F4E54).littleEndian, as: UInt32.self)
            ptr.storeBytes(of: frameNumber.littleEndian, toByteOffset: 4, as: UInt32.self)
            ptr.storeBytes(of: UInt16(metas.count).littleEndian, toByteOffset: 8, as: UInt16.self)
            ptr.storeBytes(of: UInt16(0).littleEndian, toByteOffset: 10, as: UInt16.self)

            // Descriptors + point data
            var pointOffset: UInt32 = 0
            for (i, meta) in metas.enumerated() {
                let descOff = headerSize + i * 20
                ptr.storeBytes(of: UInt16(meta.points.count).littleEndian,
                               toByteOffset: descOff, as: UInt16.self)
                ptr.storeBytes(of: meta.parentIndex.littleEndian,
                               toByteOffset: descOff + 2, as: Int16.self)
                ptr.storeBytes(of: pointOffset.littleEndian,
                               toByteOffset: descOff + 4, as: UInt32.self)
                ptr.storeBytes(of: meta.centroidX.bitPattern.littleEndian,
                               toByteOffset: descOff + 8, as: UInt32.self)
                ptr.storeBytes(of: meta.centroidY.bitPattern.littleEndian,
                               toByteOffset: descOff + 12, as: UInt32.self)
                ptr.storeBytes(of: meta.area.bitPattern.littleEndian,
                               toByteOffset: descOff + 16, as: UInt32.self)

                // Write remapped + Y-flipped points
                let pointBase = headerSize + descSize + Int(pointOffset)
                for j in 0..<meta.points.count {
                    let p = meta.points[j]
                    let rx = p.x * scaleX - offX
                    let ry = 1.0 - (p.y * scaleY - offY)
                    ptr.storeBytes(of: rx.bitPattern.littleEndian,
                                   toByteOffset: pointBase + j * 8, as: UInt32.self)
                    ptr.storeBytes(of: ry.bitPattern.littleEndian,
                                   toByteOffset: pointBase + j * 8 + 4, as: UInt32.self)
                }
                pointOffset += UInt32(meta.points.count * 8)
            }
        }

        // Send as binary WebSocket message to all connected clients
        let metadata = NWProtocolWebSocket.Metadata(opcode: .binary)
        let context = NWConnection.ContentContext(identifier: "contour",
                                                   metadata: [metadata])
        lock.withLock {
            for conn in connections {
                conn.send(content: data, contentContext: context, completion: .contentProcessed { _ in })
            }
        }
    }

    func stop() {
        listener.cancel()
        lock.withLock {
            for conn in connections { conn.cancel() }
            connections.removeAll()
        }
    }
}
