import ArgumentParser
import Foundation
import MLX
import MoebiusMLX

/// Parity gates for the Moebius Swift port, run as an EXECUTABLE.
///
/// Not an XCTest on purpose: the SPM test host cannot reliably resolve the mlx-swift metallib
/// (`native` build system ships none and aborts the whole process), while `swift run` does real
/// GPU inference. Build with `--build-system swiftbuild`.
@main
struct MoebiusGate: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "moebius-gate",
        abstract: "Parity gates for the Moebius Swift port.")

    @Option(help: "Fixture bundle written by moebius-m0/rung/parity_lambda.py.")
    var fixture: String = "/Volumes/Satechi/Development/mlxengine-image/WIP/moebius-m0/goldens/lambda/lambda_fixture.safetensors"

    @Flag(help: "Gate the LλMI block (self + cross) against the PyTorch reference.")
    var lambdaGate: Bool = false

    @Option(help: "Gate a DepthwiseSeparableConv block fixture (path to its .safetensors).")
    var convGate: String?

    @Option(help: "Gate a DWResnetBlock2D fixture (path to its .safetensors).")
    var resnetGate: String?

    @Option(help: "Gate a GLUMBConv (FFN) fixture (path to its .safetensors).")
    var ffnGate: String?

    @Option(help: "Gate a MixTransformerBlock fixture (path to its .safetensors).")
    var blockGate: String?

    @Option(help: "Gate a MixTransformer2DModel (attentions.N) fixture.")
    var attnGate: String?

    @Option(help: "Gate a DWMixTFDownBlock2D (down_blocks.N) fixture.")
    var downGate: String?

    @Option(help: "Override GroupNorm epsilon. Omit to use the value the port determined by measurement (see DWResnetBlock2D). A CLI default here would SHADOW the port's own and silently gate the wrong value.")
    var normEps: Float?

    @Flag(inversion: .prefixedNo,
          help: "Pin to the CPU stream — GPU fp32 noise (~8e-4/op) both hides and mimics op bugs. Use --no-cpu for a GPU smoke.")
    var cpu: Bool = true

    mutating func run() throws {
        if cpu { Device.setDefault(device: Device(.cpu)) }
        var ran = false
        if lambdaGate { try runLambdaGate(); ran = true }
        if let convGate { try runConvGate(path: convGate); ran = true }
        if let resnetGate { try runResnetGate(path: resnetGate); ran = true }
        if let ffnGate { try runFFNGate(path: ffnGate); ran = true }
        if let blockGate { try runBlockGate(path: blockGate); ran = true }
        if let attnGate { try runAttnGate(path: attnGate); ran = true }
        if let downGate { try runDownGate(path: downGate); ran = true }
        if !ran { print("nothing to do — pass --lambda-gate, --conv-gate or --resnet-gate") }
    }

    private func loadFixture(_ path: String) throws -> ([String: MLXArray], [String: MLXArray]) {
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MoebiusError.fixtureNotFound(url.path)
        }
        let bundle = try MLX.loadArrays(url: url)
        var weights: [String: MLXArray] = [:]
        for (k, v) in bundle where k.hasPrefix("w.") { weights[String(k.dropFirst(2))] = v }
        print("\n\(url.deletingPathExtension().lastPathComponent)")
        return (weights, bundle)
    }

    /// Gate a `MixTransformer2DModel` — norm / proj_in / blocks / proj_out / residual, spatial I/O.
    private func runAttnGate(path: String) throws {
        let (weights, bundle) = try loadFixture(path)
        guard let x = bundle["io.input0"], let ctx = bundle["io.kw_encoder_hidden_states"],
              let ref = bundle["io.output"] else {
            throw MoebiusError.missingWeight("io.input0 / io.kw_encoder_hidden_states / io.output")
        }
        let model = try MixTransformer2DModel(weights)
        let out = Layout.nhwcToNCHW(model(Layout.nchwToNHWC(x), context: ctx))
        eval(out)
        let failures = report("transformer2d", out, ref)
        print(failures == 0 ? "ATTN GATE: PASS" : "ATTN GATE: FAIL")
        if failures > 0 { throw ExitCode.failure }
    }

    /// Gate a whole down block — [resnet -> attention] x N, then downsample.
    private func runDownGate(path: String) throws {
        let (weights, bundle) = try loadFixture(path)
        guard let x = bundle["io.kw_hidden_states"], let temb = bundle["io.kw_temb"],
              let ctx = bundle["io.kw_encoder_hidden_states"], let ref = bundle["io.output"] else {
            throw MoebiusError.missingWeight("io.kw_hidden_states / io.kw_temb / io.kw_encoder_hidden_states / io.output")
        }
        let block = try DWMixTFDownBlock2D(weights)
        let (h, skips) = block(Layout.nchwToNHWC(x), temb: temb, context: ctx)
        let out = Layout.nhwcToNCHW(h)
        eval(out)
        print("  layers=\(block.resnets.count) skips=\(skips.count) downsampler=\(block.downsampler != nil)")
        let failures = report("down block", out, ref)
        print(failures == 0 ? "DOWN GATE: PASS" : "DOWN GATE: FAIL")
        if failures > 0 { throw ExitCode.failure }
    }

    /// Gate a whole `MixTransformerBlock` — the first COMPOSITE gate, exercising norm1/attn1,
    /// norm2/attn2 and norm3/ff together with their residuals.
    private func runBlockGate(path: String) throws {
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MoebiusError.fixtureNotFound(url.path)
        }
        let bundle = try MLX.loadArrays(url: url)
        var weights: [String: MLXArray] = [:]
        for (k, v) in bundle where k.hasPrefix("w.") { weights[String(k.dropFirst(2))] = v }
        guard let x = bundle["io.input0"], let ctx = bundle["io.kw_encoder_hidden_states"],
              let ref = bundle["io.output"] else {
            throw MoebiusError.missingWeight("io.input0 / io.kw_encoder_hidden_states / io.output")
        }
        print("\n\(url.deletingPathExtension().lastPathComponent)  x \(x.shape) ctx \(ctx.shape) -> \(ref.shape)")
        let block = try MixTransformerBlock(weights)
        let out = block(x, context: ctx)
        eval(out)
        let failures = report("mix transformer", out, ref)
        print(failures == 0 ? "BLOCK GATE: PASS" : "BLOCK GATE: FAIL")
        if failures > 0 { throw ExitCode.failure }
    }

    /// Gate one `GLUMBConv` FFN. Its I/O is sequence-form [b, n, c], so no layout transpose.
    private func runFFNGate(path: String) throws {
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MoebiusError.fixtureNotFound(url.path)
        }
        let bundle = try MLX.loadArrays(url: url)
        var weights: [String: MLXArray] = [:]
        for (k, v) in bundle where k.hasPrefix("w.") { weights[String(k.dropFirst(2))] = v }
        guard let x = bundle["io.input0"], let ref = bundle["io.output"] else {
            throw MoebiusError.missingWeight("io.input0 / io.output")
        }
        print("\n\(url.deletingPathExtension().lastPathComponent)  \(x.shape) -> \(ref.shape)")
        let block = try GLUMBConv(weights)
        let out = block(x)
        eval(out)
        let failures = report("glu mbconv", out, ref)
        print(failures == 0 ? "FFN GATE: PASS" : "FFN GATE: FAIL")
        if failures > 0 { throw ExitCode.failure }
    }

    /// Gate one `DWResnetBlock2D` against its captured (hidden_states, temb) → output.
    private func runResnetGate(path: String) throws {
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MoebiusError.fixtureNotFound(url.path)
        }
        let bundle = try MLX.loadArrays(url: url)
        var weights: [String: MLXArray] = [:]
        for (k, v) in bundle where k.hasPrefix("w.") { weights[String(k.dropFirst(2))] = v }
        guard let xNCHW = bundle["io.input0"], let temb = bundle["io.input1"],
              let refNCHW = bundle["io.output"] else {
            throw MoebiusError.missingWeight("io.input0 / io.input1 / io.output")
        }
        print("\n\(url.deletingPathExtension().lastPathComponent)  x \(xNCHW.shape) temb \(temb.shape) -> \(refNCHW.shape)   [eps \(normEps.map(String.init(describing:)) ?? "port default")]")

        let block: DWResnetBlock2D
        if let normEps {
            block = try DWResnetBlock2D(weights, eps: normEps)
        } else {
            block = try DWResnetBlock2D(weights)   // the port's measured value
        }
        let out = Layout.nhwcToNCHW(block(Layout.nchwToNHWC(xNCHW), temb: temb))
        eval(out)
        let failures = report("dw resnet", out, refNCHW)
        print(failures == 0 ? "RESNET GATE: PASS" : "RESNET GATE: FAIL")
        if failures > 0 { throw ExitCode.failure }
    }

    /// Gate one `DepthwiseSeparableConv` (conv_in / conv_out) against its captured I/O.
    private func runConvGate(path: String) throws {
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MoebiusError.fixtureNotFound(url.path)
        }
        let bundle = try MLX.loadArrays(url: url)
        var weights: [String: MLXArray] = [:]
        for (k, v) in bundle where k.hasPrefix("w.") { weights[String(k.dropFirst(2))] = v }
        guard let inNCHW = bundle["io.input"], let refNCHW = bundle["io.output"] else {
            throw MoebiusError.missingWeight("io.input / io.output")
        }
        print("\n\(url.deletingPathExtension().lastPathComponent)  in \(inNCHW.shape) -> out \(refNCHW.shape)")

        let block = try DepthwiseSeparableConv(weights)
        let out = Layout.nhwcToNCHW(block(Layout.nchwToNHWC(inNCHW)))
        eval(out)
        let failures = report("dwsep conv", out, refNCHW)
        print(failures == 0 ? "CONV GATE: PASS" : "CONV GATE: FAIL")
        if failures > 0 { throw ExitCode.failure }
    }

    private func runLambdaGate() throws {
        let url = URL(fileURLWithPath: fixture)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MoebiusError.fixtureNotFound(url.path)
        }
        let bundle = try MLX.loadArrays(url: url)
        print("fixture: \(bundle.count) arrays from \(url.lastPathComponent)")

        // Weights are stored with a `w.` prefix; strip it to get the reference's own key names.
        var weights: [String: MLXArray] = [:]
        for (k, v) in bundle where k.hasPrefix("w.") {
            weights[String(k.dropFirst(2))] = v
        }
        func need(_ k: String) throws -> MLXArray {
            guard let v = bundle[k] else { throw MoebiusError.missingWeight(k) }
            return v
        }
        func w(_ k: String) throws -> MLXArray {
            guard let v = weights[k] else { throw MoebiusError.missingWeight(k) }
            return v
        }

        // The reference works in NCHW; our blocks take NHWC. The fixture was saved in the layout
        // the reference's own module consumes, which for this block is ALREADY [b, hh, ww, c]
        // (it transposes internally), so no permute is needed on the way in or out.
        let x = try need("in.x")
        let ctx = try need("in.ctx")
        let refSelf = try need("ref.self_out")
        let refCross = try need("ref.cross_out")

        var failures = 0

        // ── self-attention: LOCAL branch (posConv, r = 15) ───────────────────────────────────
        let selfBlock = MultiQuerySelfLambda(
            toQ: try w("attn1.to_q.weight"),
            toK: try w("attn1.to_k.weight"),
            toV: try w("attn1.to_v.weight"),
            normQ: try BatchNormParams(weights, "attn1.norm_q"),
            normV: try BatchNormParams(weights, "attn1.norm_v"),
            posConvWeight: try w("attn1.pos_conv.weight"),
            posConvBias: try w("attn1.pos_conv.bias"))
        let outSelf = selfBlock(x)
        eval(outSelf)
        failures += report("self  (local, r=15)", outSelf, refSelf)

        // ── cross-attention: GLOBAL branch (relPosEmb, m = 10) ───────────────────────────────
        let crossBlock = MultiQueryCrossLambda(
            toQ: try w("attn2.to_q.weight"),
            toK: try w("attn2.to_k.weight"),
            toV: try w("attn2.to_v.weight"),
            normQ: try BatchNormParams(weights, "attn2.norm_q"),
            normV: try BatchNormParams(weights, "attn2.norm_v"),
            relPosEmb: try w("attn2.rel_pos_emb"))
        let outCross = crossBlock(x, hiddenStates: ctx)
        eval(outCross)
        failures += report("cross (global, m=10)", outCross, refCross)

        print(failures == 0 ? "\nLAMBDA GATE: PASS" : "\nLAMBDA GATE: \(failures) FAILURE(S)")
        if failures > 0 { throw ExitCode.failure }
    }

    /// Thresholds per the porting skill: single layer < 1e-4 ideal, < 1e-3 acceptable.
    private func report(_ name: String, _ got: MLXArray, _ want: MLXArray) -> Int {
        guard got.shape == want.shape else {
            print("  \(name): SHAPE \(got.shape) vs \(want.shape)  [FAIL]")
            return 1
        }
        let diff = MLX.abs(got - want)
        let maxAbs = diff.max().item(Float.self)
        // Scale both to ~unit magnitude before reducing. Summing squares of 1.3M fp32 values
        // straight accumulates enough error to print a cosine ABOVE 1, which reads as a broken
        // gate even when the port is exact. max|Δ| is the threshold; the cosine is context.
        let scale = MLX.maximum(MLX.abs(want).max(), MLXArray(Float(1e-20)))
        let a = got / scale, bN = want / scale
        let dot = (a * bN).sum().item(Float.self)
        let norm = sqrt(a.square().sum().item(Float.self)) * sqrt(bN.square().sum().item(Float.self))
        let cos = norm > 0 ? min(dot / norm, 1.0) : 0

        // Gate on RELATIVE-to-range, never absolute. This block's outputs span ~±140, so an
        // absolute 1e-4 threshold is meaningful on the CPU stream and meaningless on GPU, where
        // fp32 accumulation alone contributes ~8e-4 relative per the porting skill. Judging the GPU
        // run by the CPU's absolute number reports a FAIL on an exact port — the calibration trap.
        let range = (want.max() - want.min()).item(Float.self)
        let rel = range > 0 ? maxAbs / range : maxAbs
        let tolerance: Float = cpu ? 1e-6 : 5e-3
        let verdict = rel < tolerance ? "PASS" : (rel < tolerance * 10 ? "ok" : "FAIL")
        print(String(format: "  %-22s max|Δ|=%.3e  rel=%.3e  cos=%.9f   [%@ vs %@ tol %.0e]",
                     (name as NSString).utf8String!, maxAbs, rel, cos, verdict,
                     cpu ? "cpu" : "gpu", tolerance))
        return verdict == "FAIL" ? 1 : 0
    }
}
