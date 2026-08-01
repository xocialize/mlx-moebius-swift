import Foundation
import MLX

/// The Moebius inpainting pipeline — a transcription of
/// `removal/v1_2/pipeline.py::RemovalSDXLPipeline_BatchMode._denoise_steps`.
///
///     encode(image) ─┐
///     encode(masked)─┼→ [noisy(4) | mask(1) | masked(4)] → UNet ─CFG→ DDIM ─×19→ decode → paste
///     mask ──────────┘
///
/// Three things here are unusual enough to restate, because each is the kind of detail that yields
/// a plausible-but-wrong image rather than an obvious failure:
///
/// 1. **It does not start from pure noise.** `strength = 0.99` noises the CLEAN image latents at
///    `timesteps[0]`, so a trace of the original leaks into the initialisation — and 20 requested
///    steps run **19**, starting at t=900.
/// 2. **CFG is batch-doubled through ONE forward**, `[uncond, cond]`, where the two halves differ
///    only in which 10 of the 20 learned category embeddings they index.
/// 3. **The channel order is noisy(4) + mask(1) + masked(4)** — the mask sits in the MIDDLE.
public struct MoebiusPipeline {
    public let unet: MoebiusUNet
    public let vae: AutoencoderKL
    public var scheduler: DDIMScheduler

    public var numSteps: Int = 20
    public var strength: Float = 0.99
    public var guidanceScale: Float = 2.5      // the reference CLI default (its README says 2.0)
    public var noiseOffset: Float = 0.0357
    public var numEmbeddings: Int = 20

    public init(unet: MoebiusUNet, vae: AutoencoderKL) {
        self.unet = unet
        self.vae = vae
        self.scheduler = DDIMScheduler()
    }

    /// Latents for the denoise loop. Supplying `injected` replays the oracle's own VAE draw, which
    /// is how cross-framework parity is gated — `encode().sample()` is stochastic and no two
    /// frameworks draw the same noise.
    public struct Inputs {
        public var latents: MLXArray            // [1, 4, 64, 64] NCHW, already × scalingFactor
        public var maskedLatents: MLXArray      // [1, 4, 64, 64] NCHW
        public var maskLatent: MLXArray         // [1, 1, 64, 64] NCHW, 1 = remove
        public var noise: MLXArray              // [1, 4, 64, 64] NCHW, offset already folded in

        public init(latents: MLXArray, maskedLatents: MLXArray,
                    maskLatent: MLXArray, noise: MLXArray) {
            self.latents = latents
            self.maskedLatents = maskedLatents
            self.maskLatent = maskLatent
            self.noise = noise
        }
    }

    /// Run the denoise loop and decode. Returns the image in `[0, 1]`, NCHW.
    public func run(_ inputs: Inputs, progress: ((Int, Int) -> Void)? = nil) -> MLXArray {
        var sched = scheduler
        sched.setTimesteps(numSteps)
        sched.applyStrength(strength)
        let timesteps = sched.timesteps

        var noisy = sched.addNoise(inputs.latents, noise: inputs.noise, timestep: timesteps[0])

        // [uncond, cond]: the SECOND half of the 20-vector table is the unconditional branch.
        let half = numEmbeddings / 2
        let uncond = MLXArray(Int32(half) ..< Int32(numEmbeddings)).expandedDimensions(axis: 0)
        let cond = MLXArray(Int32(0) ..< Int32(half)).expandedDimensions(axis: 0)
        let inputIds = concatenated([uncond, cond], axis: 0)

        let maskPair = concatenated([inputs.maskLatent, inputs.maskLatent], axis: 0)
        let maskedPair = concatenated([inputs.maskedLatents, inputs.maskedLatents], axis: 0)

        for (i, t) in timesteps.enumerated() {
            let scaled = sched.scaleModelInput(noisy, timestep: t)
            let pair = concatenated([scaled, scaled], axis: 0)
            // noisy(4) | mask(1) | masked(4) — the mask is the MIDDLE channel group.
            let modelInput = concatenated([pair, maskPair, maskedPair], axis: 1)

            let timestepArray = MLXArray([Int32(t), Int32(t)])
            let predNCHW = Layout.nhwcToNCHW(
                unet(Layout.nchwToNHWC(modelInput), timestep: timestepArray, inputIds: inputIds))

            let predUncond = predNCHW[0 ..< 1]
            let predCond = predNCHW[1 ..< 2]
            let predCFG = predUncond + (predCond - predUncond) * guidanceScale

            noisy = sched.step(modelOutput: predCFG, timestep: t, sample: noisy)
            // Per-step eval: 19 chained lazy UNet graphs would blow the Metal command-buffer
            // timeout and retain every intermediate.
            eval(noisy)
            progress?(i + 1, timesteps.count)
        }

        let decoded = (vae.decode(noisy / vae.scalingFactor) + 1) / 2
        eval(decoded)
        return decoded
    }

    /// `_post_process` with `paste = true`: composite through a Gaussian-blurred mask so only the
    /// erased region comes from the model. The blur is what keeps the seam from being visible —
    /// and it is also why judging quality means judging the GENERATED region, not the pasted whole.
    public static func paste(result: MLXArray, source: MLXArray, mask: MLXArray,
                             blurRadius: Int = 3) -> MLXArray {
        var soft = mask
        for _ in 0 ..< blurRadius { soft = boxBlur3NCHW(soft) }
        return result * soft + source * (1 - soft)
    }

    static func boxBlur3NCHW(_ x: MLXArray) -> MLXArray {
        let h = x.dim(2), w = x.dim(3)
        let padV = concatenated([x[0..., 0..., 0 ..< 1, 0...], x,
                                 x[0..., 0..., (h - 1) ..< h, 0...]], axis: 2)
        let p = concatenated([padV[0..., 0..., 0..., 0 ..< 1], padV,
                              padV[0..., 0..., 0..., (w - 1) ..< w]], axis: 3)
        var acc = p[0..., 0..., 0 ..< h, 0 ..< w]
        for dy in 0 ..< 3 {
            for dx in 0 ..< 3 where !(dy == 0 && dx == 0) {
                acc = acc + p[0..., 0..., dy ..< (dy + h), dx ..< (dx + w)]
            }
        }
        return acc / 9
    }
}
