import Foundation
import MLX

/// SANA's `GLUMBConv` — the feed-forward network inside every MixTransformer block.
/// Reference: `model_lib/nets/layers/sana/basic_modules.py`.
///
/// An inverted-bottleneck MBConv with a gated-linear-unit split:
///
///     [B,N,C] → (B,H,W,C)
///       → inverted_conv 1×1  C → 2·hidden   (+bias, then **SiLU**)
///       → depth_conv    3×3  depthwise      (+bias, **no activation**)
///       → chunk into (value, gate)          — FIRST half is the value
///       → value * SiLU(gate)
///       → point_conv    1×1  hidden → C     (**no bias, no activation**)
///     → [B,N,C]
///
/// The asymmetric `use_bias=(True, True, False)` and `act=("silu","silu",None)` are the details to
/// get right — point_conv has neither bias nor activation, and the depthwise stage has a bias but
/// no activation. Moebius sets `mix_mlp_ratio = 2.5`, so at C=320: hidden = 800 and the inverted
/// conv emits 1600.
public struct GLUMBConv {
    public let invertedConvWeight: MLXArray   // MLX [2·hidden, 1, 1, C]
    public let invertedConvBias: MLXArray
    public let depthConvWeight: MLXArray      // MLX [2·hidden, k, k, 1]
    public let depthConvBias: MLXArray
    public let pointConvWeight: MLXArray      // MLX [C, 1, 1, hidden]
    public let padding: Int

    public init(_ weights: [String: MLXArray], prefix: String = "") throws {
        let p = prefix.isEmpty ? "" : "\(prefix)."
        func need(_ k: String) throws -> MLXArray {
            guard let v = weights["\(p)\(k)"] else { throw MoebiusError.missingWeight("\(p)\(k)") }
            return v
        }
        let inv = try need("inverted_conv.conv.weight")
        self.invertedConvWeight = inv.transposed(0, 2, 3, 1)
        self.invertedConvBias = try need("inverted_conv.conv.bias")
        let dw = try need("depth_conv.conv.weight")
        self.depthConvWeight = dw.transposed(0, 2, 3, 1)
        self.depthConvBias = try need("depth_conv.conv.bias")
        let pw = try need("point_conv.conv.weight")
        self.pointConvWeight = pw.transposed(0, 2, 3, 1)
        self.padding = dw.dim(2) / 2
    }

    /// - Parameter x: `[b, n, c]` sequence form, with `n = h·w` and h == w (the block is square).
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), n = x.dim(1), c = x.dim(2)
        let h = Int(Double(n).squareRoot())

        var t = x.reshaped([b, h, h, c])                       // sequence → spatial (NHWC)
        t = conv2d(t, invertedConvWeight, padding: IntOrPair(0)) + invertedConvBias
        t = siluActivation(t)
        let twoHidden = t.shape.last!
        t = conv2d(t, depthConvWeight, padding: IntOrPair(padding), groups: twoHidden)
            + depthConvBias                                    // depthwise, no activation

        // `x, gate = torch.chunk(x, 2, dim=1)` on NCHW — the channel axis. Channels are last here.
        let hidden = twoHidden / 2
        let value = t[0..., 0..., 0..., 0 ..< hidden]
        let gate = t[0..., 0..., 0..., hidden ..< twoHidden]
        t = value * siluActivation(gate)

        t = conv2d(t, pointConvWeight, padding: IntOrPair(0))  // no bias, no activation
        return t.reshaped([b, n, c])
    }
}
