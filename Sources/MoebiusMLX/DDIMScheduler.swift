import Foundation
import MLX

/// DDIM, configured exactly as Moebius' `build_pipeline` does:
/// `beta_start 0.00085`, `beta_end 0.012`, `scaled_linear`, 1000 train steps, `clip_sample false`.
///
/// The sampler is a first-class port surface, not an afterthought — a plausible-looking substitute
/// passes latent-cosine checks and still produces wrong images. This is a direct transcription of
/// diffusers' `DDIMScheduler`, restricted to the settings Moebius actually uses:
/// `eta = 0` (deterministic, no added noise), `prediction_type = "epsilon"`, `clip_sample = false`,
/// `set_alpha_to_one = true` (so the final `alphaProdPrev` is exactly 1).
public struct DDIMScheduler {
    public let alphasCumprod: [Float]
    public let numTrainTimesteps: Int
    public private(set) var timesteps: [Int] = []
    public private(set) var numInferenceSteps: Int = 0

    public init(betaStart: Float = 0.00085, betaEnd: Float = 0.012,
                numTrainTimesteps: Int = 1000) {
        self.numTrainTimesteps = numTrainTimesteps
        // "scaled_linear": linear in SQRT space, then squared — not a linear beta ramp.
        var cumulative: Float = 1
        var acc: [Float] = []
        acc.reserveCapacity(numTrainTimesteps)
        let lo = betaStart.squareRoot(), hi = betaEnd.squareRoot()
        for i in 0 ..< numTrainTimesteps {
            let t = numTrainTimesteps == 1 ? 0 : Float(i) / Float(numTrainTimesteps - 1)
            let beta = { let s = lo + (hi - lo) * t; return s * s }()
            cumulative *= (1 - beta)
            acc.append(cumulative)
        }
        self.alphasCumprod = acc
    }

    /// `[950, 900, …, 0]` for 20 steps over 1000 train steps — descending, `step_ratio` apart.
    public mutating func setTimesteps(_ steps: Int) {
        numInferenceSteps = steps
        let ratio = numTrainTimesteps / steps
        timesteps = (0 ..< steps).map { $0 * ratio }.reversed()
    }

    /// The pipeline's `strength = 0.99` drops the leading timestep, leaving 19 of 20.
    public mutating func applyStrength(_ strength: Float) {
        let initSteps = min(Int(Float(numInferenceSteps) * strength), numInferenceSteps)
        let start = max(numInferenceSteps - initSteps, 0)
        timesteps = Array(timesteps.dropFirst(start))
    }

    /// `q(x_t | x_0)` — used once, to noise the CLEAN latents at the first timestep.
    public func addNoise(_ original: MLXArray, noise: MLXArray, timestep: Int) -> MLXArray {
        let a = alphasCumprod[timestep]
        return original * a.squareRoot() + noise * (1 - a).squareRoot()
    }

    /// One deterministic DDIM step (eta = 0), epsilon-prediction.
    public func step(modelOutput: MLXArray, timestep: Int, sample: MLXArray) -> MLXArray {
        let prev = timestep - numTrainTimesteps / numInferenceSteps
        let alphaProd = alphasCumprod[timestep]
        // set_alpha_to_one = true: past the start of the schedule, alpha_prod_prev is exactly 1.
        let alphaProdPrev = prev >= 0 ? alphasCumprod[prev] : Float(1)
        let betaProd = 1 - alphaProd

        // clip_sample = false, so pred_original_sample is used unclamped.
        let predOriginal = (sample - modelOutput * betaProd.squareRoot()) / alphaProd.squareRoot()
        let direction = modelOutput * (1 - alphaProdPrev).squareRoot()
        return predOriginal * alphaProdPrev.squareRoot() + direction
    }

    /// DDIM does not rescale its input; kept so the pipeline reads 1:1 against the reference.
    public func scaleModelInput(_ sample: MLXArray, timestep: Int) -> MLXArray { sample }
}
