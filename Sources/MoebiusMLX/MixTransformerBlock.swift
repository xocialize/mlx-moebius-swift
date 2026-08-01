import Foundation
import MLX

/// `MixTransformerBlock` — diffusers' `BasicTransformerBlock` with the feed-forward swapped for
/// SANA's `GLUMBConv`, and both attentions replaced by LλMI. Reference:
/// `model_lib/nets/layers/unet_blocks/mix_transformer.py` + diffusers `BasicTransformerBlock`.
///
/// `norm_type == "layer_norm"` (the default), so the forward is plain pre-norm residual:
///
///     h = h + attn1(norm1(h))                 — self  (LOCAL λ)
///     h = h + attn2(norm2(h), context)        — cross (GLOBAL λ)
///     h = h + ff(norm3(h))                    — GLUMBConv
///
/// Everything is sequence-form `[b, n, c]`; the λ blocks are spatial, so each is wrapped in a
/// reshape that recovers `h = w = √n` — the same `MQSλ_FwdWrapper` / `MQXλ_FwdWrapper` the
/// reference uses, and the reason the model cannot accept a non-square latent.
public struct MixTransformerBlock {
    public let norm1: LayerNormParams
    public let attn1: MultiQuerySelfLambda
    public let norm2: LayerNormParams
    public let attn2: MultiQueryCrossLambda
    public let norm3: LayerNormParams
    public let ff: GLUMBConv
    public let eps: Float

    public init(_ weights: [String: MLXArray], prefix: String = "",
                heads: Int = 8, dimU: Int = 1, r: Int = 15, eps: Float = 1e-5) throws {
        let p = prefix.isEmpty ? "" : "\(prefix)."
        func need(_ k: String) throws -> MLXArray {
            guard let v = weights["\(p)\(k)"] else { throw MoebiusError.missingWeight("\(p)\(k)") }
            return v
        }
        self.norm1 = try LayerNormParams(weights, "\(p)norm1")
        self.norm2 = try LayerNormParams(weights, "\(p)norm2")
        self.norm3 = try LayerNormParams(weights, "\(p)norm3")
        self.attn1 = MultiQuerySelfLambda(
            toQ: try need("attn1.to_q.weight"),
            toK: try need("attn1.to_k.weight"),
            toV: try need("attn1.to_v.weight"),
            normQ: try BatchNormParams(weights, "\(p)attn1.norm_q"),
            normV: try BatchNormParams(weights, "\(p)attn1.norm_v"),
            posConvWeight: try need("attn1.pos_conv.weight"),
            posConvBias: try need("attn1.pos_conv.bias"),
            heads: heads, dimU: dimU, r: r)
        self.attn2 = MultiQueryCrossLambda(
            toQ: try need("attn2.to_q.weight"),
            toK: try need("attn2.to_k.weight"),
            toV: try need("attn2.to_v.weight"),
            normQ: try BatchNormParams(weights, "\(p)attn2.norm_q"),
            normV: try BatchNormParams(weights, "\(p)attn2.norm_v"),
            relPosEmb: try need("attn2.rel_pos_emb"),
            heads: heads, dimU: dimU)
        self.ff = try GLUMBConv(weights, prefix: prefix.isEmpty ? "ff" : "\(p)ff")
        self.eps = eps
    }

    /// - Parameters:
    ///   - x: `[b, n, c]` sequence form
    ///   - context: `[b, m, dimCross]` — the 10 projected category embeddings
    public func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        var h = x
        h = h + attn1.sequence(LayerNormParams.apply(h, norm1, eps: eps))
        h = h + attn2.sequence(LayerNormParams.apply(h, norm2, eps: eps), context: context)
        h = h + ff(LayerNormParams.apply(h, norm3, eps: eps))
        return h
    }
}

extension MultiQuerySelfLambda {
    /// `MQSλ_FwdWrapper`: `[b, n, c] → [b, √n, √n, c] → block → [b, n, c]`.
    public func sequence(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), n = x.dim(1), c = x.dim(2)
        let s = Int(Double(n).squareRoot())
        return self(x.reshaped([b, s, s, c])).reshaped([b, n, c])
    }
}

extension MultiQueryCrossLambda {
    /// `MQXλ_FwdWrapper`: same reshape sandwich, plus the cross context.
    public func sequence(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let b = x.dim(0), n = x.dim(1), c = x.dim(2)
        let s = Int(Double(n).squareRoot())
        return self(x.reshaped([b, s, s, c]), hiddenStates: context).reshaped([b, n, c])
    }
}

/// LayerNorm over the last axis.
public struct LayerNormParams {
    public let weight: MLXArray
    public let bias: MLXArray

    public init(_ weights: [String: MLXArray], _ prefix: String) throws {
        guard let w = weights["\(prefix).weight"], let b = weights["\(prefix).bias"] else {
            throw MoebiusError.missingWeight("\(prefix).weight/bias")
        }
        self.weight = w
        self.bias = b
    }

    public static func apply(_ x: MLXArray, _ p: LayerNormParams, eps: Float) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = x.variance(axis: -1, keepDims: true)
        return (x - mean) * rsqrt(variance + eps) * p.weight + p.bias
    }
}
