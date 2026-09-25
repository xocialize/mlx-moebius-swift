// Weight-free probe: mlx's Winograd conv2d path at Moebius shapes, raw vs the routed conv
// (Sources/MoebiusMLX/WinogradFreeConv2d.swift), both against the CPU conv — including the UNet's
// `up_blocks.1` upsampler as production runs it (fp16 weights, fp32 activations, batch 2).
//
// Measured 2026-09-24 on the M5 Max, mlx-swift 0.31.6: raw conv2d ~6.4e-3 relL2 at every shape
// below (Winograd), route ~1e-6. Removal signal: raw conv2d reported exact on a new pin.
//
// Run: swift test --filter WinogradProbeTests

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import MoebiusMLX

final class WinogradProbeTests: XCTestCase {

    static func relL2(_ a: MLXArray, _ b: MLXArray) -> (rel: Float, maxAbs: Float) {
        let d = a.asType(.float32) - b.asType(.float32)
        let rel = sqrt(sum(d * d)) / sqrt(sum(square(b.asType(.float32))))
        let mx = abs(d).max()
        eval(rel, mx)
        return (rel.item(Float.self), mx.item(Float.self))
    }

    func testMoebiusConvShapes() throws {
        // (N, C, O, spatial, weight dtype): VAE encoder/decoder stages, then the UNet upsampler.
        let cases: [(Int, Int, Int, Int, DType)] = [
            (1, 512, 512, 64, .float16), (1, 256, 256, 128, .float16), (1, 128, 128, 256, .float16),
            (2, 640, 640, 64, .float16),
        ]
        for (n, c, o, s, wdt) in cases {
            MLXRandom.seed(UInt64(c * 7 + o + n))
            let x = MLXRandom.normal([n, s, s, c])
            let w = (MLXRandom.normal([o, 3, 3, c]) * (1.0 / Float(c * 9).squareRoot())).asType(wdt)
            let b = MLXRandom.normal([o]).asType(wdt) * 0.1
            XCTAssertTrue(
                WinogradFreeConv2d.takesWinograd(
                    input: x, weight: w, stride: (1, 1), dilation: (1, 1), groups: 1))
            let ref = Device.withDefaultDevice(.cpu) { () -> MLXArray in
                let r = conv2d(x, w.asType(.float32), stride: 1, padding: 1) + b.asType(.float32)
                eval(r)
                return r
            }
            let raw = WinogradFreeConv2d.conv(x, weight: w, bias: b, padding: (1, 1), route: .winograd)
            let routed = WinogradFreeConv2d.conv(x, weight: w, bias: b, padding: (1, 1), route: .conv3d)
            eval(raw, routed)
            let r0 = Self.relL2(raw, ref), r1 = Self.relL2(routed, ref)
            print(String(format: "  N%d %d→%d @%d² (w %@): raw conv2d relL2 %.2e (max %.1e) %@ | route relL2 %.2e",
                         n, c, o, s, String(describing: wdt), r0.rel, r0.maxAbs,
                         r0.rel < 1e-5 ? "EXACT — route removable" : "lossy", r1.rel))
            XCTAssertLessThan(r1.rel, 1e-5, "conv3d route must be exact-class")
        }
    }
}
