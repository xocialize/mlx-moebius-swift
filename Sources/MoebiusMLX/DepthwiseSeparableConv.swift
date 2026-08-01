import Foundation
import MLX

/// timm's `DepthwiseSeparableConv` (MobileNet-V1 lineage), as Moebius uses it for `conv_in` and
/// `conv_out`. Reference: `model_lib/nets/layers/_efficientnet_blocks.py`.
///
/// Moebius constructs it with ALL defaults — `DWConv2d(in, out, dw_kernel_size=k)` — which resolves
/// to a configuration worth stating explicitly, because three of these are the "constructor
/// defaults" trap and none are visible at the call site:
///   • `act_layer = nn.ReLU`   ← **ReLU, not SiLU**, despite the UNet's top-level `conv_act`
///   • `group_size = 1`        → `groups = in_chs`, i.e. FULLY depthwise
///   • `pw_act = False`        → `bn2` applies NO activation (while `bn1` does)
///   • `se_layer = None`, `aa_layer = None` → `se` and `aa` are Identity
///   • `has_skip = (stride == 1 && in == out)` → FALSE for both 9→320 and 320→4
///
/// forward: `conv_dw → bn1+ReLU → aa → se → conv_pw → bn2`
///
/// PERFORMANCE NOTE (not a correctness note). Depthwise-separable convolution is a *mobile-NPU*
/// efficiency trick: it maps to efficient primitives on ANE/Hexagon, but on MLX/Metal it is
/// memory- and launch-bound, and measured FLOPs reduction does not predict speedup — see memory
/// `mlx-no-grouped-conv3d`, where a 20–25× FLOPs cut bought 0.57–2.17× wall-clock and was once
/// SLOWER than the dense conv it replaced. Grouped conv2d does work (only grouped *conv3d* is
/// missing), so this is correct here; but Moebius' headline speed claim was measured on CUDA and
/// should be re-measured on Metal rather than assumed.
public struct DepthwiseSeparableConv {
    public let convDW: MLXArray      // MLX layout [inChs, kH, kW, 1]
    public let bn1: BatchNormParams
    public let convPW: MLXArray      // MLX layout [outChs, 1, 1, inChs]
    public let bn2: BatchNormParams
    public let groups: Int
    public let padding: Int

    /// Build from a flat weight dict holding the REFERENCE's PyTorch layouts. The conv transposes
    /// happen here — the same conversion the production weight loader performs — so the gate
    /// exercises that logic rather than being handed pre-transposed tensors.
    public init(_ weights: [String: MLXArray], prefix: String = "") throws {
        let p = prefix.isEmpty ? "" : "\(prefix)."
        func need(_ k: String) throws -> MLXArray {
            guard let v = weights["\(p)\(k)"] else { throw MoebiusError.missingWeight("\(p)\(k)") }
            return v
        }
        // PyTorch conv weight is (O, I/groups, kH, kW); MLX wants (O, kH, kW, I/groups).
        let dw = try need("conv_dw.weight")
        self.convDW = dw.transposed(0, 2, 3, 1)
        let pw = try need("conv_pw.weight")
        self.convPW = pw.transposed(0, 2, 3, 1)
        self.bn1 = try BatchNormParams(weights, "\(p)bn1")
        self.bn2 = try BatchNormParams(weights, "\(p)bn2")
        self.groups = dw.dim(0)                 // fully depthwise: groups == in_chs
        self.padding = dw.dim(2) / 2            // timm resolves '' to 'same' for odd kernels
    }

    /// - Parameter x: `[b, h, w, inChs]` (NHWC). Returns `[b, h, w, outChs]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv2d(x, convDW, padding: IntOrPair(padding), groups: groups)
        h = Lambda.batchNormInference(h, weight: bn1.weight, bias: bn1.bias,
                                      runningMean: bn1.runningMean, runningVar: bn1.runningVar)
        h = maximum(h, 0)                       // bn1 is a norm_act layer: BatchNorm + ReLU
        // aa and se are Identity under Moebius' construction — kept as comments rather than
        // no-op code so the forward reads 1:1 against the reference.
        h = conv2d(h, convPW, padding: IntOrPair(0))
        h = Lambda.batchNormInference(h, weight: bn2.weight, bias: bn2.bias,
                                      runningMean: bn2.runningMean, runningVar: bn2.runningVar)
        return h                                // bn2 applies no activation (pw_act == false)
    }
}

/// Layout helpers. The reference and every golden are NCHW; MLX is NHWC.
public enum Layout {
    @inlinable public static func nchwToNHWC(_ x: MLXArray) -> MLXArray { x.transposed(0, 2, 3, 1) }
    @inlinable public static func nhwcToNCHW(_ x: MLXArray) -> MLXArray { x.transposed(0, 3, 1, 2) }
}
