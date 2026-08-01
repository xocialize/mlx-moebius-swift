import Foundation
import MLX

/// Sinusoidal timestep features — diffusers `Timesteps` / `get_timestep_embedding`.
///
/// UNet2DConditionModel defaults apply (`flip_sin_to_cos = true`, `freq_shift = 0`), and the
/// flip is not cosmetic: it swaps the sin and cos halves, so getting it wrong shifts every
/// timestep's identity while still producing a smooth, plausible embedding.
public enum Timesteps {
    public static func embedding(_ timesteps: MLXArray, dim: Int,
                                 flipSinToCos: Bool = true, downscaleFreqShift: Float = 0,
                                 maxPeriod: Float = 10000, scale: Float = 1) -> MLXArray {
        let half = dim / 2
        let exponent = -MLX.log(MLXArray(maxPeriod))
            * MLXArray(0 ..< half).asType(.float32) / (Float(half) - downscaleFreqShift)
        let freqs = MLX.exp(exponent)                                   // [half]
        var emb = timesteps.asType(.float32).expandedDimensions(axis: 1) * freqs
            .expandedDimensions(axis: 0)                                // [b, half]
        emb = emb * scale
        emb = concatenated([MLX.sin(emb), MLX.cos(emb)], axis: -1)      // [b, dim]
        if flipSinToCos {
            emb = concatenated([emb[0..., half...], emb[0..., ..<half]], axis: -1)
        }
        return emb
    }
}

/// `TimestepEmbedding` — linear_1 → SiLU → linear_2.
public struct TimestepEmbedding {
    public let linear1: LinearParams
    public let linear2: LinearParams

    public init(_ weights: [String: MLXArray], prefix: String) throws {
        self.linear1 = try LinearParams(weights, "\(prefix).linear_1")
        self.linear2 = try LinearParams(weights, "\(prefix).linear_2")
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var t = matmul(x, linear1.weight.transposed(1, 0)) + linear1.bias
        t = siluActivation(t)
        return matmul(t, linear2.weight.transposed(1, 0)) + linear2.bias
    }
}

/// The Moebius UNet — `UNet2DLambdaDWConvMixFFNConditionModel_prune_down_mid_up_block_8x8`.
///
///     conv_in → [down_blocks] → [up_blocks] → conv_norm_out → SiLU → conv_out
///
/// **NO mid block** (`mid_block_type: null`), which is unusual for a diffusers UNet and is why the
/// down path ends and the up path begins with no bottleneck between them.
///
/// SKIP BOOKKEEPING is the subtle part. `down_block_res_samples` is seeded with **conv_in's own
/// output** before any down block runs, so the totals are 1 + 3 + 3 + 2 = 9 skips against 3 up
/// blocks × 3 resnets = 9 consumers. Each up block takes the LAST `resnets.count` skips and the
/// list is shortened from the end.
public struct MoebiusUNet {
    public let convIn: DepthwiseSeparableConv
    public let timeEmbedding: TimestepEmbedding
    public let encoderHidProj: LinearParams
    public let downBlocks: [DWMixTFDownBlock2D]
    public let upBlocks: [DWMixTFUpBlock2D]
    public let convNormOut: GroupNormParams
    public let convOut: DepthwiseSeparableConv
    public let embeddingLayer: MLXArray          // [numEmbeddings, encoderHidDim]
    public let timeProjDim: Int
    public let groups: Int
    public let eps: Float

    /// - Parameter weights: the full checkpoint, in PyTorch layouts, keys unchanged.
    public init(_ weights: [String: MLXArray], groups: Int = 32, eps: Float = 1e-5,
                timeProjDim: Int = 320, heads: Int = 8) throws {
        let d = "diff_model."
        self.convIn = try DepthwiseSeparableConv(weights, prefix: "\(d)conv_in")
        self.convOut = try DepthwiseSeparableConv(weights, prefix: "\(d)conv_out")
        self.timeEmbedding = try TimestepEmbedding(weights, prefix: "\(d)time_embedding")
        self.encoderHidProj = try LinearParams(weights, "\(d)encoder_hid_proj")
        self.convNormOut = try GroupNormParams(weights, "\(d)conv_norm_out")
        guard let emb = weights["embedding_layer.weight"] else {
            throw MoebiusError.missingWeight("embedding_layer.weight")
        }
        self.embeddingLayer = emb

        var downs: [DWMixTFDownBlock2D] = []
        var i = 0
        while weights.keys.contains(where: { $0.hasPrefix("\(d)down_blocks.\(i).") }) {
            downs.append(try DWMixTFDownBlock2D(weights, prefix: "\(d)down_blocks.\(i)",
                                                heads: heads, resnetEps: eps))
            i += 1
        }
        var ups: [DWMixTFUpBlock2D] = []
        i = 0
        while weights.keys.contains(where: { $0.hasPrefix("\(d)up_blocks.\(i).") }) {
            ups.append(try DWMixTFUpBlock2D(weights, prefix: "\(d)up_blocks.\(i)",
                                            heads: heads, resnetEps: eps))
            i += 1
        }
        self.downBlocks = downs
        self.upBlocks = ups
        self.groups = groups
        self.eps = eps
        self.timeProjDim = timeProjDim
    }

    /// - Parameters:
    ///   - sample: `[b, 64, 64, 9]` NHWC — noisy(4) + mask(1) + masked(4)
    ///   - timestep: `[b]`
    ///   - inputIds: `[b, m]` — indices into the 20-vector category table
    public func callAsFunction(_ sample: MLXArray, timestep: MLXArray,
                               inputIds: MLXArray) -> MLXArray {
        // Conditioning: gather the category embeddings, then project 3072 → 768.
        var context = embeddingLayer[inputIds]                         // [b, m, 3072]
        context = matmul(context, encoderHidProj.weight.transposed(1, 0)) + encoderHidProj.bias

        let tEmb = Timesteps.embedding(timestep, dim: timeProjDim)
        let emb = timeEmbedding(tEmb)

        var h = convIn(sample)
        var skips: [MLXArray] = [h]                                    // seeded with conv_in output
        for block in downBlocks {
            let (out, produced) = block(h, temb: emb, context: context)
            h = out
            skips.append(contentsOf: produced)
        }

        for block in upBlocks {
            let take = block.resnets.count
            let slice = Array(skips.suffix(take))
            skips.removeLast(take)
            h = block(h, skips: slice, temb: emb, context: context)
        }

        h = GroupNormParams.apply(h, convNormOut, groups: groups, eps: eps)
        h = siluActivation(h)
        return convOut(h)
    }
}
