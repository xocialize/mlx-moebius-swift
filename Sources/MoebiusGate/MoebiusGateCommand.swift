import ArgumentParser
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXNN
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

    @Option(help: "Gate a DWMixTFUpBlock2D (up_blocks.N) fixture.")
    var upGate: String?

    @Flag(help: "Gate the WHOLE UNet against step0_pred_raw.")
    var unetGate: Bool = false

    @Flag(help: "Gate the DDIM scheduler: timestep schedule, add_noise, and one step.")
    var ddimGate: Bool = false

    @Flag(help: "Gate the VAE encoder (posterior mean) and decoder.")
    var vaeGate: Bool = false

    @Flag(help: "Run the FULL pipeline against the oracle's own latents and compare decoded images.")
    var pipelineGate: Bool = false

    @Option(help: "Write the pipeline's decoded image here (PNG).")
    var outputImage: String?

    @Option(help: "Converted VAE (safetensors).")
    var vaeCheckpoint: String = "/Volumes/Satechi/Development/mlxengine-image/WIP/moebius-m0/converted/moebius-vae-fp32.safetensors"

    @Option(help: "Converted checkpoint (safetensors) for the UNet gate.")
    var checkpoint: String = "/Volumes/Satechi/Development/mlxengine-image/WIP/moebius-m0/converted/moebius-ft_places2-fp32.safetensors"

    @Option(help: "Pipeline golden bundle for the UNet gate.")
    var unetFixture: String = "/Volumes/Satechi/Development/mlxengine-image/WIP/moebius-m0/goldens/unet_fixture.safetensors"

    @Option(help: "Cast model weights to this dtype before running (fp16|bf16). Omit for the checkpoint's own fp32. BatchNorm running stats stay fp32 regardless — a running_var that rounds toward zero makes rsqrt explode.")
    var modelDtype: String?

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
        if let upGate { try runUpGate(path: upGate); ran = true }
        if unetGate { try runUNetGate(); ran = true }
        if ddimGate { try runDDIMGate(); ran = true }
        if vaeGate { try runVAEGate(); ran = true }
        if pipelineGate { try runPipelineGate(); ran = true }
        if !ran { print("nothing to do — pass --lambda-gate, --conv-gate or --resnet-gate") }
    }

    /// Cast weights for a dtype lane. BatchNorm statistics are pinned fp32 (see convert_weights.py);
    /// everything else float takes the requested dtype. Inputs are cast by the callers so the conv
    /// stacks genuinely COMPUTE in the low dtype rather than promoting straight back to fp32.
    private func applyDtype(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        guard let modelDtype else { return weights }
        let dtype: DType = modelDtype == "bf16" ? .bfloat16 : .float16
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (k, v) in weights {
            let pinned = k.hasSuffix("running_mean") || k.hasSuffix("running_var")
            out[k] = (!pinned && v.dtype == .float32) ? v.asType(dtype) : v
        }
        return out
    }

    private var lowDtype: DType? {
        guard let modelDtype else { return nil }
        return modelDtype == "bf16" ? .bfloat16 : .float16
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

    /// The decisive gate: the whole pipeline, 19 steps, against the oracle's decoded image.
    ///
    /// The oracle's VAE draw and noise are INJECTED so the two runs are comparable — otherwise the
    /// stochastic `encode().sample()` alone would make them different (valid) images.
    private func runPipelineGate() throws {
        let started = Date()
        let unetWeights = try MLX.loadArrays(url: URL(fileURLWithPath: checkpoint))
        let vaeWeights = try MLX.loadArrays(url: URL(fileURLWithPath: vaeCheckpoint))
        let fx = try MLX.loadArrays(url: URL(fileURLWithPath: unetFixture))
        guard let latents = fx["io.latents"], let maskedLatents = fx["io.masked_latents"],
              let maskLatent = fx["io.resized_masks"], let noise = fx["io.noise"],
              let ref = fx["io.decoded"] else {
            throw MoebiusError.missingWeight("pipeline goldens")
        }
        let vae = AutoencoderKL()
        try vae.update(parameters: ModuleParameters.unflattened(AutoencoderKL.sanitize(vaeWeights)),
                       verify: [.all])
        eval(vae)
        let pipeline = MoebiusPipeline(unet: try MoebiusUNet(applyDtype(unetWeights)), vae: vae)
        print("\npipeline  cfg=\(pipeline.guidanceScale) strength=\(pipeline.strength) steps=\(pipeline.numSteps) dtype=\(modelDtype ?? "fp32")")

        // The scheduler state stays fp32 (matching the reference, whose DDIM math is fp32 even in
        // fp16 runs); only the UNet weights + its input are low-dtype. The pipeline casts the model
        // input per step via the UNet weights' dtype — here we pre-cast the conditioning latents.
        let inputs = MoebiusPipeline.Inputs(latents: latents, maskedLatents: maskedLatents,
                                            maskLatent: maskLatent, noise: noise)
        let image = pipeline.run(inputs) { step, total in
            FileHandle.standardError.write(Data("  step \(step)/\(total)\r".utf8))
        }
        FileHandle.standardError.write(Data("\n".utf8))

        // A 19-step trajectory AMPLIFIES per-step precision differences, so pixel-comparing a GPU
        // run against a CPU-computed golden is the wrong gate — the same reason the porting skill
        // says not to gate quantized generative models on PSNR-vs-golden: the run lands on a
        // different but equally valid image. Measured here: GPU vs the CPU golden is rel 2.5e-01
        // with cos 0.99986, and the two images are visually indistinguishable.
        //
        // CHASED, NOT ASSUMED (2026-08-01). Two probes rule out a GPU-specific defect:
        //   1. ONE UNet forward on GPU vs the CPU golden: rel 1.671e-03, cos 0.9999998 — inside
        //      the known fp32 accumulation envelope (~8e-4/op through a 226M stack). The 19-step
        //      drift is therefore iteration of an in-envelope per-step delta, not a bad kernel.
        //   2. Two GPU pipeline runs are BIT-IDENTICAL to each other — fully deterministic, so
        //      nothing nondeterministic is misfiring; the CPU/GPU gap is systematic rounding only.
        // The NAX split-K GEMM registry bug (mlx#3797) does not apply: it needs half-precision
        // with K >= 10240, and this pipeline is fp32 with max K = 1600.
        //   • CPU lane  -> pixel parity against the oracle (the real correctness gate)
        //   • GPU lane  -> structural agreement (cosine) + a visual sample
        let failures: Int
        if cpu {
            failures = report("decoded image", image, ref, tolerance: VAE_FP32_TOLERANCE)
        } else {
            let a = image.reshaped([-1]), b = ref.reshaped([-1])
            let scale = MLX.maximum(MLX.abs(b).max(), MLXArray(Float(1e-20)))
            let an = a / scale, bn = b / scale
            let cos = (an * bn).sum().item(Float.self)
                / (sqrt(an.square().sum().item(Float.self)) * sqrt(bn.square().sum().item(Float.self)))
            let ok = cos > 0.999
            print(String(format: "  decoded image (GPU)    cos=%.9f   [%@ — structural gate; pixel parity is CPU-lane only]",
                         cos, ok ? "PASS" : "FAIL"))
            failures = ok ? 0 : 1
        }
        print(String(format: "  %.1f s for %d steps", Date().timeIntervalSince(started), 19))
        if let outputImage {
            try writePNG(image, to: URL(fileURLWithPath: outputImage))
            print("  wrote \(outputImage)")
        }
        print(failures == 0 ? "PIPELINE GATE: PASS" : "PIPELINE GATE: FAIL")
        if failures > 0 { throw ExitCode.failure }
    }

    /// Write an NCHW [0,1] image as PNG.
    private func writePNG(_ imageNCHW: MLXArray, to url: URL) throws {
        let h = imageNCHW.dim(2), w = imageNCHW.dim(3)
        let hwc = MLX.clip(imageNCHW[0].transposed(1, 2, 0) * 255, min: 0, max: 255).asType(.uint8)
        eval(hwc)
        let bytes: [UInt8] = hwc.asArray(UInt8.self)
        var rgba = [UInt8](repeating: 255, count: w * h * 4)
        for i in 0 ..< (w * h) {
            rgba[i * 4] = bytes[i * 3]; rgba[i * 4 + 1] = bytes[i * 3 + 1]
            rgba[i * 4 + 2] = bytes[i * 3 + 2]
        }
        let provider = CGDataProvider(data: Data(rgba) as CFData)!
        let cg = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                         bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                         bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                         provider: provider, decode: nil, shouldInterpolate: false,
                         intent: .defaultIntent)!
        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw MoebiusError.fixtureNotFound(url.path)
        }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { throw MoebiusError.fixtureNotFound(url.path) }
    }

    /// The VAE's fp32 accumulation floor, MEASURED rather than chosen.
    ///
    /// The rest of this port gates at 1e-6 relative, but that was calibrated on 64x64 UNet tensors.
    /// The VAE runs at 512x512 with up to 512 channels — ~64x more accumulation per output — and
    /// fp32 simply cannot hold 1e-6 there. Rather than loosen the bound by feel, the floor was
    /// measured with the self-control the porting skill prescribes: PyTorch AGAINST ITSELF at
    /// fp32 vs fp64 on the identical computation gives rel 2.840e-04 (encoder mean) and 8.064e-06
    /// (decoder). Our Swift-vs-PyTorch numbers are 3.871e-04 and 1.332e-05 — the same order, within
    /// ~1.5x of the framework's own precision floor. A real port bug would not sit there.
    private let VAE_FP32_TOLERANCE: Float = 1e-3

    /// Gate the VAE. The encoder is gated on the posterior MEAN — the reference calls `.sample()`,
    /// whose noise draw no other framework reproduces — and the decoder on the final image.
    private func runVAEGate() throws {
        let url = URL(fileURLWithPath: vaeCheckpoint)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MoebiusError.fixtureNotFound(url.path)
        }
        let weights = try MLX.loadArrays(url: url)
        let fx = try MLX.loadArrays(url: URL(fileURLWithPath: unetFixture))
        guard let image = fx["io.image"], let latentsMean = fx["io.latents_mean"],
              let latentsFinal = fx["io.latents_final"], let decodedRef = fx["io.decoded"] else {
            throw MoebiusError.missingWeight("vae goldens")
        }
        let vae = AutoencoderKL()
        let params = ModuleParameters.unflattened(AutoencoderKL.sanitize(weights))
        try vae.update(parameters: params, verify: [.all])
        eval(vae)
        print("\nVAE  \(weights.count) tensors, scalingFactor=\(vae.scalingFactor)")

        var failures = 0
        // ENCODER: posterior mean, scaled the way the pipeline scales it.
        let mean = vae.encode(image).mean * vae.scalingFactor
        eval(mean)
        failures += report("encoder mean", mean, latentsMean, tolerance: VAE_FP32_TOLERANCE)

        // DECODER: the pipeline decodes latents/scalingFactor then maps [-1,1] -> [0,1].
        let decoded = (vae.decode(latentsFinal / vae.scalingFactor) + 1) / 2
        eval(decoded)
        failures += report("decoder", decoded, decodedRef, tolerance: VAE_FP32_TOLERANCE)

        print(failures == 0 ? "VAE GATE: PASS" : "VAE GATE: FAIL")
        if failures > 0 { throw ExitCode.failure }
    }

    /// Gate DDIM: the schedule itself, add_noise, and one deterministic step.
    private func runDDIMGate() throws {
        let fx = try MLX.loadArrays(url: URL(fileURLWithPath: unetFixture))
        guard let latents = fx["io.latents"], let noise = fx["io.noise"],
              let noisyInitial = fx["io.noisy_initial"], let predCFG = fx["io.step0_pred_cfg"],
              let latentsOut = fx["io.step0_latents_out"], let t = fx["io.timestep"] else {
            throw MoebiusError.missingWeight("ddim goldens")
        }
        var sched = DDIMScheduler()
        sched.setTimesteps(20)
        sched.applyStrength(0.99)
        let first = sched.timesteps[0]
        let expected = Int(t[0].item(Int32.self))
        print("\nDDIM  steps=\(sched.timesteps.count) first=\(first) (oracle timestep \(expected))")
        var failures = first == expected ? 0 : 1
        if first != expected { print("  SCHEDULE MISMATCH  [FAIL]") }

        let noised = sched.addNoise(latents, noise: noise, timestep: first)
        eval(noised)
        failures += report("add_noise", noised, noisyInitial)

        let stepped = sched.step(modelOutput: predCFG, timestep: first, sample: noisyInitial)
        eval(stepped)
        failures += report("one DDIM step", stepped, latentsOut)

        print(failures == 0 ? "DDIM GATE: PASS" : "DDIM GATE: FAIL")
        if failures > 0 { throw ExitCode.failure }
    }

    /// Gate the WHOLE UNet: 9 skips, 3 down blocks, 3 up blocks, no mid block.
    private func runUNetGate() throws {
        let ckptURL = URL(fileURLWithPath: checkpoint)
        guard FileManager.default.fileExists(atPath: ckptURL.path) else {
            throw MoebiusError.fixtureNotFound(ckptURL.path)
        }
        let started = Date()
        let weights = try MLX.loadArrays(url: ckptURL)
        print("\ncheckpoint: \(weights.count) tensors from \(ckptURL.lastPathComponent)")

        let fx = try MLX.loadArrays(url: URL(fileURLWithPath: unetFixture))
        guard let input = fx["io.step0_model_input"], let t = fx["io.timestep"],
              let ids = fx["io.input_ids"], let ref = fx["io.step0_pred_raw"] else {
            throw MoebiusError.missingWeight("io.step0_model_input / io.timestep / io.input_ids / io.step0_pred_raw")
        }
        let unet = try MoebiusUNet(applyDtype(weights))
        print("  down=\(unet.downBlocks.count) up=\(unet.upBlocks.count) timestep=\(t[0].item(Int32.self)) dtype=\(modelDtype ?? "fp32")")

        var netInput = Layout.nchwToNHWC(input)
        if let lowDtype { netInput = netInput.asType(lowDtype) }
        let out = Layout.nhwcToNCHW(unet(netInput, timestep: t, inputIds: ids)).asType(.float32)
        eval(out)
        let failures = report("UNet e2e", out, ref)
        print(String(format: "  %.1f s", Date().timeIntervalSince(started)))
        print(failures == 0 ? "UNET GATE: PASS" : "UNET GATE: FAIL")
        if failures > 0 { throw ExitCode.failure }
    }

    /// Gate a whole up block — skip concat (LIFO) -> resnet -> attention, then upsample.
    private func runUpGate(path: String) throws {
        let (weights, bundle) = try loadFixture(path)
        guard let x = bundle["io.kw_hidden_states"], let temb = bundle["io.kw_temb"],
              let ctx = bundle["io.kw_encoder_hidden_states"], let ref = bundle["io.output"] else {
            throw MoebiusError.missingWeight("io.kw_hidden_states / io.kw_temb / io.kw_encoder_hidden_states / io.output")
        }
        // Skips arrive as kw_res_hidden_states_tuple_<i>, in production order.
        var skips: [MLXArray] = []
        var i = 0
        while let s = bundle["io.kw_res_hidden_states_tuple_\(i)"] {
            skips.append(Layout.nchwToNHWC(s)); i += 1
        }
        let block = try DWMixTFUpBlock2D(weights)
        print("  layers=\(block.resnets.count) skips=\(skips.count) upsampler=\(block.upsampler != nil)")
        let out = Layout.nhwcToNCHW(block(Layout.nchwToNHWC(x), skips: skips, temb: temb, context: ctx))
        eval(out)
        let failures = report("up block", out, ref)
        print(failures == 0 ? "UP GATE: PASS" : "UP GATE: FAIL")
        if failures > 0 { throw ExitCode.failure }
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
    private func report(_ name: String, _ got: MLXArray, _ want: MLXArray,
                        tolerance override: Float? = nil) -> Int {
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
        let tolerance: Float = override ?? (cpu ? 1e-6 : 5e-3)
        let verdict = rel < tolerance ? "PASS" : (rel < tolerance * 10 ? "ok" : "FAIL")
        print(String(format: "  %-22s max|Δ|=%.3e  rel=%.3e  cos=%.9f   [%@ vs %@ tol %.0e]",
                     (name as NSString).utf8String!, maxAbs, rel, cos, verdict,
                     cpu ? "cpu" : "gpu", tolerance))
        return verdict == "FAIL" ? 1 : 0
    }
}
