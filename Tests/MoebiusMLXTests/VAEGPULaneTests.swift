// Moebius VAE (SDXL KL-f8, fp16 weights × fp32 activations as in production) on the GPU lane
// (conv route on / raw Winograd) vs the CPU lane, on a real DIV2K photo at the fixed 512²
// processing size, plus the timing of both routes. MoebiusGate runs on the CPU stream by default,
// so the GPU lane's Winograd loss was never visible there.
//
// Run: MOEBIUS_PARITY=1 MOEBIUS_DIR=<dir with vae.safetensors> \
//      swift test -c release -Xswiftc -enable-testing --filter MoebiusMLXTests.VAEGPULaneTests
// Override: MOEBIUS_REAL_IMAGE.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXNN
import XCTest

@testable import MoebiusMLX

final class VAEGPULaneTests: XCTestCase {
    static let env = ProcessInfo.processInfo.environment
    static let realImage = URL(
        fileURLWithPath: env["MOEBIUS_REAL_IMAGE"]
            ?? "/Volumes/Satechi/Development/mlxengine-image/corpus/sr-bench/DIV2K_valid_HR/0801.png")

    static func stats(_ a: MLXArray, _ ref: MLXArray) -> String {
        let d = a.asType(.float32) - ref.asType(.float32)
        let rel = sqrt(sum(d * d)) / sqrt(sum(square(ref.asType(.float32))))
        let mx = abs(d).max()
        let mse = mean(d * d)
        eval(rel, mx, mse)
        let psnr = 10 * log10(4 / max(mse.item(Float.self), 1e-30))
        return String(format: "relL2 %.2e  maxAbs %.2e  PSNR %6.2f dB", rel.item(Float.self), mx.item(Float.self), psnr)
    }

    static func onCPU(_ f: () -> MLXArray) -> MLXArray {
        Device.withDefaultDevice(.cpu) { () -> MLXArray in
            let r = f()
            eval(r)
            return r
        }
    }

    static func gpuRun(_ vae: AutoencoderKL, route: MoebiusConvRoute, reps: Int = 3, _ f: () -> MLXArray)
        -> (MLXArray, Double)
    {
        let saved = (vae.encoderConvRoute, vae.decoderConvRoute)
        vae.encoderConvRoute = route
        vae.decoderConvRoute = route
        defer { (vae.encoderConvRoute, vae.decoderConvRoute) = saved }
        var out = f()
        eval(out)
        let t0 = Date()
        for _ in 0..<reps {
            out = f()
            eval(out)
        }
        return (out, Date().timeIntervalSince(t0) / Double(reps) * 1000)
    }

    static func loadCrop(_ url: URL, side: Int) throws -> MLXArray {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
            let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw NSError(domain: "MoebiusVAE", code: 1) }
        let (w, h) = (cg.width, cg.height)
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let ctx = CGContext(
            data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        let (x0, y0) = ((w - side) / 2, (h - side) / 2)
        let plane = side * side
        var chw = [Float](repeating: 0, count: 3 * plane)
        for y in 0..<side {
            for x in 0..<side {
                let p = ((y0 + y) * w + (x0 + x)) * 4
                for c in 0..<3 { chw[c * plane + y * side + x] = Float(rgba[p + c]) / 127.5 - 1 }
            }
        }
        return MLXArray(chw, [1, 3, side, side])
    }

    func testRealPhoto512() throws {
        try XCTSkipUnless(Self.env["MOEBIUS_PARITY"] == "1", "set MOEBIUS_PARITY=1 to run")
        let dir = URL(fileURLWithPath: try XCTUnwrap(Self.env["MOEBIUS_DIR"], "set MOEBIUS_DIR"))
        let vae = AutoencoderKL()
        try vae.update(
            parameters: ModuleParameters.unflattened(
                AutoencoderKL.sanitize(try MLX.loadArrays(url: dir.appendingPathComponent("vae.safetensors")))),
            verify: [.all])
        eval(vae)
        let pixels = try Self.loadCrop(Self.realImage, side: 512)
        let mCPU = Self.onCPU { vae.encode(pixels).mean }
        let (mR, teR) = Self.gpuRun(vae, route: .conv3d) { vae.encode(pixels).mean }
        let (mW, teW) = Self.gpuRun(vae, route: .winograd) { vae.encode(pixels).mean }
        let z = mCPU
        let ref = Self.onCPU { vae.decode(z) }
        Memory.clearCache()
        let (dR, tdR) = Self.gpuRun(vae, route: .conv3d) { vae.decode(z) }
        let (dW, tdW) = Self.gpuRun(vae, route: .winograd) { vae.decode(z) }
        print("[real 512² photo: \(Self.realImage.lastPathComponent)]")
        print(String(format: "  encode mean  conv3d %@  %6.1f ms", Self.stats(mR, mCPU), teR))
        print(String(format: "  encode mean  raw    %@  %6.1f ms", Self.stats(mW, mCPU), teW))
        print(String(format: "  decode       conv3d %@  %6.1f ms", Self.stats(dR, ref), tdR))
        print(String(format: "  decode       raw    %@  %6.1f ms", Self.stats(dW, ref), tdW))
    }
}
