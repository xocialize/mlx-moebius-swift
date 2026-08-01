import Foundation
import MLX

/// `DWResnetBlock2D` — Moebius' residual block. Reference:
/// `model_lib/nets/layers/unet_blocks/dw_resnet.py`.
///
/// Standard diffusers `ResnetBlock2D` shape with the two 3×3 convs swapped for
/// `DepthwiseSeparableConv`:
///
///     norm1 → SiLU → conv1 → (+ temb) → norm2 → SiLU → conv2 → (x + h) / outputScaleFactor
///
/// GroupNorm eps is **1e-5**, DETERMINED BY MEASUREMENT rather than read off a default. The block's
/// own signature says `eps: float = 1e-6`, but the diffusers UNet passes its `norm_eps` down and
/// wins: gating the same fixture at both values gives rel 6.773e-07 at 1e-5 vs 1.889e-06 at 1e-6.
/// This is the "resolved config, not the json" trap — the value in force is not the one written at
/// the definition site.
///
/// ⚠️ TWO DIFFERENT ACTIVATIONS live in this block. The resnet's own `nonlinearity` is **SiLU**
/// (`non_linearity="swish"`), while the `bn1` inside each `DepthwiseSeparableConv` applies **ReLU**
/// (timm's `act_layer` default). Collapsing them to one activation is a silent-output bug.
/// SiLU / swish. Defined here rather than pulled from MLXNN so the core math stays explicit —
/// and so it cannot be confused with the ReLU inside `DepthwiseSeparableConv`.
@inlinable public func siluActivation(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

public struct DWResnetBlock2D {
    public let norm1: GroupNormParams
    public let conv1: DepthwiseSeparableConv
    public let timeEmbProj: LinearParams
    public let norm2: GroupNormParams
    public let conv2: DepthwiseSeparableConv
    public let convShortcut: LinearParams?     // 1×1 conv, present only when in != out
    public let outputScaleFactor: Float
    public let groups: Int
    public let eps: Float

    public init(_ weights: [String: MLXArray], prefix: String = "",
                groups: Int = 32, eps: Float = 1e-5, outputScaleFactor: Float = 1.0) throws {
        let p = prefix.isEmpty ? "" : "\(prefix)."
        self.norm1 = try GroupNormParams(weights, "\(p)norm1")
        self.norm2 = try GroupNormParams(weights, "\(p)norm2")
        self.conv1 = try DepthwiseSeparableConv(weights, prefix: "\(p)conv1")
        self.conv2 = try DepthwiseSeparableConv(weights, prefix: "\(p)conv2")
        self.timeEmbProj = try LinearParams(weights, "\(p)time_emb_proj")
        self.convShortcut = try? LinearParams(weights, "\(p)conv_shortcut")
        self.groups = groups
        self.eps = eps
        self.outputScaleFactor = outputScaleFactor
    }

    /// - Parameters:
    ///   - x: `[b, h, w, c]` (NHWC)
    ///   - temb: `[b, tembDim]`
    public func callAsFunction(_ x: MLXArray, temb: MLXArray) -> MLXArray {
        var h = GroupNormParams.apply(x, norm1, groups: groups, eps: eps)
        h = siluActivation(h)
        h = conv1(h)

        // Reference: `temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]`.
        // In NCHW that appends two spatial axes; in NHWC the channel axis is last, so the two new
        // axes go in the MIDDLE. Expressed with expandedDimensions rather than a subscript — a
        // transcribed `[:, None]` can silently no-op in Swift-MLX (memory
        // `mlx-swift-newaxis-subscript-trap`).
        let t = matmul(siluActivation(temb), timeEmbProj.weight.transposed(1, 0)) + timeEmbProj.bias
        h = h + t.expandedDimensions(axes: [1, 2])          // [b, 1, 1, c]

        h = GroupNormParams.apply(h, norm2, groups: groups, eps: eps)
        h = siluActivation(h)
        // dropout is Identity at inference
        h = conv2(h)

        var residual = x
        if let convShortcut {
            // 1×1 conv == channel matmul.
            residual = matmul(x, convShortcut.weight.squeezed(axes: [2, 3]).transposed(1, 0))
            if let b = convShortcut.biasOrNil { residual = residual + b }
        }
        return (residual + h) / outputScaleFactor
    }
}

/// GroupNorm parameters + the NHWC normalisation.
public struct GroupNormParams {
    public let weight: MLXArray
    public let bias: MLXArray

    public init(_ weights: [String: MLXArray], _ prefix: String) throws {
        guard let w = weights["\(prefix).weight"], let b = weights["\(prefix).bias"] else {
            throw MoebiusError.missingWeight("\(prefix).weight/bias")
        }
        self.weight = w
        self.bias = b
    }

    /// GroupNorm over `[b, h, w, c]`.
    ///
    /// PyTorch groups CONSECUTIVE channels on NCHW; with channels last, reshaping
    /// `[b,h,w,c] → [b,h,w,g,c/g]` groups the same channels, and the reduction runs over
    /// (h, w, c/g) — i.e. axes 1, 2 and 4.
    public static func apply(_ x: MLXArray, _ p: GroupNormParams, groups g: Int, eps: Float)
        -> MLXArray {
        let b = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.shape.last!
        let grouped = x.reshaped([b, h, w, g, c / g])
        let mean = grouped.mean(axes: [1, 2, 4], keepDims: true)
        let variance = grouped.variance(axes: [1, 2, 4], keepDims: true)
        let normalized = (grouped - mean) * rsqrt(variance + eps)
        return normalized.reshaped([b, h, w, c]) * p.weight + p.bias
    }
}

/// A `Linear` (or 1×1 conv) with an optional bias.
public struct LinearParams {
    public let weight: MLXArray
    public let biasOrNil: MLXArray?
    public var bias: MLXArray { biasOrNil ?? MLXArray(Float(0)) }

    public init(_ weights: [String: MLXArray], _ prefix: String) throws {
        guard let w = weights["\(prefix).weight"] else {
            throw MoebiusError.missingWeight("\(prefix).weight")
        }
        self.weight = w
        self.biasOrNil = weights["\(prefix).bias"]
    }
}
