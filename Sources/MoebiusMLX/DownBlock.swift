import Foundation
import MLX

/// `MixTransformer2DModel` — diffusers' continuous-input `Transformer2DModel` wrapping N
/// `MixTransformerBlock`s. Reference: `model_lib/nets/layers/unet_blocks/mix_transformer.py`.
///
///     residual = x
///     x = norm(x) → proj_in → to sequence → blocks… → to spatial → proj_out
///     return x + residual
///
/// ⚠️ This `norm` uses **eps 1e-6**, hardcoded by diffusers at the construction site, while the
/// resnets in the very same down-block use **1e-5** (their `norm_eps` is passed down from the
/// UNet). Two different GroupNorm epsilons coexist one level apart — assuming a single value for
/// "the model's GroupNorm eps" is wrong in one of the two places.
public struct MixTransformer2DModel {
    public let norm: GroupNormParams
    public let projIn: LinearParams
    public let projOut: LinearParams
    public let blocks: [MixTransformerBlock]
    public let groups: Int
    public let eps: Float

    public init(_ weights: [String: MLXArray], prefix: String = "",
                groups: Int = 32, eps: Float = 1e-6, heads: Int = 8) throws {
        let p = prefix.isEmpty ? "" : "\(prefix)."
        self.norm = try GroupNormParams(weights, "\(p)norm")
        self.projIn = try LinearParams(weights, "\(p)proj_in")
        self.projOut = try LinearParams(weights, "\(p)proj_out")
        let count = Self.blockCount(weights, prefix: "\(p)transformer_blocks")
        var built: [MixTransformerBlock] = []
        for i in 0 ..< count {
            built.append(try MixTransformerBlock(weights, prefix: "\(p)transformer_blocks.\(i)",
                                                 heads: heads))
        }
        self.blocks = built
        self.groups = groups
        self.eps = eps
    }

    static func blockCount(_ weights: [String: MLXArray], prefix: String) -> Int {
        var n = 0
        while weights.keys.contains(where: { $0.hasPrefix("\(prefix).\(n).") }) { n += 1 }
        return n
    }

    /// - Parameter x: `[b, h, w, c]` (NHWC)
    public func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let b = x.dim(0), h = x.dim(1), w = x.dim(2)
        let residual = x

        var t = GroupNormParams.apply(x, norm, groups: groups, eps: eps)
        t = matmul(t, projIn.weight.squeezed(axes: [2, 3]).transposed(1, 0)) + projIn.bias
        let inner = t.shape.last!
        t = t.reshaped([b, h * w, inner])           // to sequence

        for block in blocks {
            t = block(t, context: context)
        }

        t = t.reshaped([b, h, w, inner])            // back to spatial
        t = matmul(t, projOut.weight.squeezed(axes: [2, 3]).transposed(1, 0)) + projOut.bias
        return t + residual
    }
}

/// diffusers `Downsample2D` — a DENSE 3×3 stride-2 conv (not depthwise, unlike everything else
/// in this model).
public struct Downsample2D {
    public let weight: MLXArray      // MLX [out, kH, kW, in]
    public let bias: MLXArray

    public init(_ weights: [String: MLXArray], prefix: String) throws {
        guard let w = weights["\(prefix).conv.weight"], let b = weights["\(prefix).conv.bias"] else {
            throw MoebiusError.missingWeight("\(prefix).conv.weight/bias")
        }
        self.weight = w.transposed(0, 2, 3, 1)
        self.bias = b
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv2d(x, weight, stride: IntOrPair(2), padding: IntOrPair(1)) + bias
    }
}

/// `DWMixTFDownBlock2D` — `[resnet → attention] × numLayers`, then an optional downsample.
///
/// Returns the hidden state AND the per-stage outputs: the UNet consumes those as skip
/// connections, and the downsampled result is appended as its own skip.
public struct DWMixTFDownBlock2D {
    public let resnets: [DWResnetBlock2D]
    public let attentions: [MixTransformer2DModel]
    public let downsampler: Downsample2D?

    public init(_ weights: [String: MLXArray], prefix: String = "",
                heads: Int = 8, resnetEps: Float = 1e-5, transformerEps: Float = 1e-6) throws {
        let p = prefix.isEmpty ? "" : "\(prefix)."
        var r: [DWResnetBlock2D] = []
        var a: [MixTransformer2DModel] = []
        var i = 0
        while weights.keys.contains(where: { $0.hasPrefix("\(p)resnets.\(i).") }) {
            r.append(try DWResnetBlock2D(weights, prefix: "\(p)resnets.\(i)", eps: resnetEps))
            a.append(try MixTransformer2DModel(weights, prefix: "\(p)attentions.\(i)",
                                               eps: transformerEps, heads: heads))
            i += 1
        }
        self.resnets = r
        self.attentions = a
        self.downsampler = try? Downsample2D(weights, prefix: "\(p)downsamplers.0")
    }

    public func callAsFunction(_ x: MLXArray, temb: MLXArray, context: MLXArray)
        -> (hidden: MLXArray, skips: [MLXArray]) {
        var h = x
        var skips: [MLXArray] = []
        for (resnet, attn) in zip(resnets, attentions) {
            h = resnet(h, temb: temb)
            h = attn(h, context: context)
            eval(h)
            skips.append(h)
        }
        if let downsampler {
            h = downsampler(h)
            eval(h)
            skips.append(h)
        }
        return (h, skips)
    }
}
