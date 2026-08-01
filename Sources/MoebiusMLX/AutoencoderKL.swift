// VENDORED, essentially unchanged, from mlx-video/PROD/musetalk-mlx-swift
// (Sources/MuseTalk/VAE.swift) — an MLX-Swift port of the diffusers SD `AutoencoderKL`, already
// PROD-validated and C14-clean there.
//
// REUSE, DON'T PORT. Moebius' VAE is the SDXL member of the same KL-f8 family: identical topology
// (blockOutChannels [128,256,512,512], latentChannels 4, DiagonalGaussian posterior) and — verified
// against our converted checkpoint — an identical KEY SET, including the NEW diffusers attention
// naming (to_q/to_k/to_v/to_out.0). The older `sd-vae-ft-mse` upload uses the legacy
// query/key/value/proj_attn spelling and differs from ours in exactly those 16 keys; this port
// already normalises to the new one, so nothing had to be remapped.
//
// The ONLY differences from the MuseTalk original:
//   • `MuseTalkConstants.vaeScalingFactor` (SD 1.5's 0.18215) -> Moebius' **0.13025** (SDXL).
//     Getting this wrong scales every latent and yields a washed-out or blown-out decode.
//   • `VAEConfig` and the eps constant are inlined here so this file stands alone.
//
// Vendored rather than depended upon: taking a package dependency on musetalk-mlx-swift would drag
// a whole talking-head model in for 261 lines of VAE. Not extracted to a shared package either —
// two consumers is the threshold for *noticing*, not for extracting (the mlx-porting rule), and
// the two differ in a constant.

import Foundation
import MLX
import MLXNN

/// diffusers VAE GroupNorm eps — 1e-6, and NOT the UNet's 1e-5. Both live in this package.
private let GN_EPS: Float = 1e-6

/// SD/SDXL KL-f8 geometry (diffusers `AutoencoderKL`).
public struct VAEConfig: Sendable {
    public var blockOutChannels: [Int] = [128, 256, 512, 512]
    public var inChannels = 3
    public var outChannels = 3
    public var latentChannels = 4
    public var layersPerBlock = 2
    public var normNumGroups = 32
    public init() {}
}

/// Moebius/PixelHacker use the SDXL VAE scaling factor, not SD 1.5's 0.18215.
public let moebiusVAEScalingFactor: Float = 0.13025

// --------------------------------------------------------------------------- //
// building blocks (keys mirror diffusers)
// --------------------------------------------------------------------------- //
final class VAEResnetBlock2D: Module {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

    init(_ inCh: Int, _ outCh: Int, groups: Int = 32) {
        self._norm1.wrappedValue = GroupNorm(groupCount: groups, dimensions: inCh, eps: GN_EPS, pytorchCompatible: true)
        self._conv1.wrappedValue = Conv2d(inputChannels: inCh, outputChannels: outCh, kernelSize: 3, padding: 1)
        self._norm2.wrappedValue = GroupNorm(groupCount: groups, dimensions: outCh, eps: GN_EPS, pytorchCompatible: true)
        self._conv2.wrappedValue = Conv2d(inputChannels: outCh, outputChannels: outCh, kernelSize: 3, padding: 1)
        self._convShortcut.wrappedValue = inCh != outCh
            ? Conv2d(inputChannels: inCh, outputChannels: outCh, kernelSize: 1) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(silu(norm1(x)))
        h = conv2(silu(norm2(h)))
        let res = convShortcut != nil ? convShortcut!(x) : x
        return res + h
    }
}

/// diffusers VAE downsample: asymmetric pad (bottom/right) then stride-2 conv, padding 0.
final class VAEDownsample2D: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(_ ch: Int) {
        self._conv.wrappedValue = Conv2d(inputChannels: ch, outputChannels: ch, kernelSize: 3, stride: 2, padding: 0)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // NHWC: pad H and W on bottom/right only.
        let padded = MLX.padded(x, widths: [IntOrPair(0), IntOrPair((0, 1)), IntOrPair((0, 1)), IntOrPair(0)])
        return conv(padded)
    }
}

/// diffusers upsample: nearest x2 then stride-1 conv padding 1.
final class VAEUpsample2D: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(_ ch: Int) {
        self._conv.wrappedValue = Conv2d(inputChannels: ch, outputChannels: ch, kernelSize: 3, padding: 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2)  // NHWC nearest-2x
        return conv(y)
    }
}

/// Single-head spatial self-attention (VAE mid block). residual_connection=True.
final class VAEAttention: Module {
    @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]   // to_out.0 (to_out.1 = dropout, omitted)

    init(_ ch: Int, groups: Int = 32) {
        self._groupNorm.wrappedValue = GroupNorm(groupCount: groups, dimensions: ch, eps: GN_EPS, pytorchCompatible: true)
        self._toQ.wrappedValue = Linear(ch, ch)
        self._toK.wrappedValue = Linear(ch, ch)
        self._toV.wrappedValue = Linear(ch, ch)
        self._toOut.wrappedValue = [Linear(ch, ch)]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let res = x
        let y = groupNorm(x).reshaped(b, h * w, c)                     // (B, HW, C)
        let q = toQ(y), k = toK(y), v = toV(y)
        let scale = 1.0 / Foundation.sqrt(Float(c))                    // dim_head = C (single head)
        let attn = softmax(q.matmul(k.transposed(0, 2, 1)) * scale, axis: -1)
        let o = toOut[0](attn.matmul(v)).reshaped(b, h, w, c)
        return o + res
    }
}

final class VAEMidBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [VAEResnetBlock2D]
    @ModuleInfo(key: "attentions") var attentions: [VAEAttention]
    init(_ ch: Int) {
        self._resnets.wrappedValue = [VAEResnetBlock2D(ch, ch), VAEResnetBlock2D(ch, ch)]
        self._attentions.wrappedValue = [VAEAttention(ch)]
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = resnets[0](x)
        h = attentions[0](h)
        h = resnets[1](h)
        return h
    }
}

final class VAEDownBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [VAEResnetBlock2D]
    @ModuleInfo(key: "downsamplers") var downsamplers: [VAEDownsample2D]?
    init(_ inCh: Int, _ outCh: Int, nRes: Int, addDownsample: Bool) {
        self._resnets.wrappedValue = (0 ..< nRes).map { VAEResnetBlock2D($0 == 0 ? inCh : outCh, outCh) }
        self._downsamplers.wrappedValue = addDownsample ? [VAEDownsample2D(outCh)] : nil
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in resnets { h = r(h) }
        if let ds = downsamplers { h = ds[0](h) }
        return h
    }
}

final class VAEUpBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [VAEResnetBlock2D]
    @ModuleInfo(key: "upsamplers") var upsamplers: [VAEUpsample2D]?
    init(_ inCh: Int, _ outCh: Int, nRes: Int, addUpsample: Bool) {
        self._resnets.wrappedValue = (0 ..< nRes).map { VAEResnetBlock2D($0 == 0 ? inCh : outCh, outCh) }
        self._upsamplers.wrappedValue = addUpsample ? [VAEUpsample2D(outCh)] : nil
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in resnets { h = r(h) }
        if let us = upsamplers { h = us[0](h) }
        return h
    }
}

// --------------------------------------------------------------------------- //
// encoder / decoder
// --------------------------------------------------------------------------- //
final class VAEEncoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "down_blocks") var downBlocks: [VAEDownBlock]
    @ModuleInfo(key: "mid_block") var midBlock: VAEMidBlock
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(_ cfg: VAEConfig) {
        let boc = cfg.blockOutChannels
        self._convIn.wrappedValue = Conv2d(inputChannels: cfg.inChannels, outputChannels: boc[0], kernelSize: 3, padding: 1)
        var blocks: [VAEDownBlock] = []
        var inCh = boc[0]
        for (i, outCh) in boc.enumerated() {
            blocks.append(VAEDownBlock(inCh, outCh, nRes: cfg.layersPerBlock, addDownsample: i != boc.count - 1))
            inCh = outCh
        }
        self._downBlocks.wrappedValue = blocks
        self._midBlock.wrappedValue = VAEMidBlock(boc[boc.count - 1])
        self._convNormOut.wrappedValue = GroupNorm(groupCount: cfg.normNumGroups, dimensions: boc[boc.count - 1], eps: GN_EPS, pytorchCompatible: true)
        self._convOut.wrappedValue = Conv2d(inputChannels: boc[boc.count - 1], outputChannels: 2 * cfg.latentChannels, kernelSize: 3, padding: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for b in downBlocks { h = b(h) }
        h = midBlock(h)
        return convOut(silu(convNormOut(h)))
    }
}

final class VAEDecoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "mid_block") var midBlock: VAEMidBlock
    @ModuleInfo(key: "up_blocks") var upBlocks: [VAEUpBlock]
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(_ cfg: VAEConfig) {
        let rev = Array(cfg.blockOutChannels.reversed())     // [512,512,256,128]
        let nRes = cfg.layersPerBlock + 1
        self._convIn.wrappedValue = Conv2d(inputChannels: cfg.latentChannels, outputChannels: rev[0], kernelSize: 3, padding: 1)
        self._midBlock.wrappedValue = VAEMidBlock(rev[0])
        var blocks: [VAEUpBlock] = []
        var inCh = rev[0]
        for (i, outCh) in rev.enumerated() {
            blocks.append(VAEUpBlock(inCh, outCh, nRes: nRes, addUpsample: i != rev.count - 1))
            inCh = outCh
        }
        self._upBlocks.wrappedValue = blocks
        self._convNormOut.wrappedValue = GroupNorm(groupCount: cfg.normNumGroups, dimensions: rev[rev.count - 1], eps: GN_EPS, pytorchCompatible: true)
        self._convOut.wrappedValue = Conv2d(inputChannels: rev[rev.count - 1], outputChannels: cfg.outChannels, kernelSize: 3, padding: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        h = midBlock(h)
        for b in upBlocks { h = b(h) }
        return convOut(silu(convNormOut(h)))
    }
}

/// Diagonal Gaussian over the encoder moments (NCHW). Not a Module — holds no parameters.
public struct DiagonalGaussian {
    public let mean: MLXArray
    public let logvar: MLXArray
    public let std: MLXArray
    init(momentsNCHW: MLXArray) {
        let parts = MLX.split(momentsNCHW, parts: 2, axis: 1)
        self.mean = parts[0]
        self.logvar = MLX.clip(parts[1], min: -30.0, max: 20.0)
        self.std = MLX.exp(0.5 * logvar)
    }
    public func sample(key: MLXArray? = nil) -> MLXArray {
        mean + std * MLXRandom.normal(mean.shape, key: key)
    }
}

// --------------------------------------------------------------------------- //
// top-level AutoencoderKL
// --------------------------------------------------------------------------- //
public final class AutoencoderKL: Module {
    @ModuleInfo(key: "encoder") var encoder: VAEEncoder
    @ModuleInfo(key: "decoder") var decoder: VAEDecoder
    @ModuleInfo(key: "quant_conv") var quantConv: Conv2d
    @ModuleInfo(key: "post_quant_conv") var postQuantConv: Conv2d

    public let scalingFactor: Float

    public init(_ cfg: VAEConfig = VAEConfig(), scalingFactor: Float = moebiusVAEScalingFactor) {
        let lc = cfg.latentChannels
        self._encoder.wrappedValue = VAEEncoder(cfg)
        self._decoder.wrappedValue = VAEDecoder(cfg)
        self._quantConv.wrappedValue = Conv2d(inputChannels: 2 * lc, outputChannels: 2 * lc, kernelSize: 1)
        self._postQuantConv.wrappedValue = Conv2d(inputChannels: lc, outputChannels: lc, kernelSize: 1)
        self.scalingFactor = scalingFactor
        super.init()
    }

    // ---- NHWC core ----
    func encodeMoments(_ xNHWC: MLXArray) -> MLXArray { quantConv(encoder(xNHWC)) }
    func decodeNHWC(_ zNHWC: MLXArray) -> MLXArray { decoder(postQuantConv(zNHWC)) }

    // ---- NCHW public API (matches diffusers/MuseTalk tensor layout) ----
    public func encode(_ xNCHW: MLXArray) -> DiagonalGaussian {
        let momentsNHWC = encodeMoments(xNCHW.transposed(0, 2, 3, 1))
        return DiagonalGaussian(momentsNCHW: momentsNHWC.transposed(0, 3, 1, 2))
    }

    public func decode(_ zNCHW: MLXArray) -> MLXArray {
        decodeNHWC(zNCHW.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
    }
}

extension AutoencoderKL {
    /// Convert a PyTorch-layout VAE state dict into MLX layout.
    ///
    /// The checkpoint converter is deliberately layout-preserving (see `convert_weights.py`), so the
    /// OIHW → OHWI transpose lives here, in the loader — one conversion path shared by the gates and
    /// production. In this VAE every 4-D tensor is a conv weight (attention q/k/v/out are `Linear`,
    /// i.e. 2-D), so the rule is simply "transpose the rank-4 ones".
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (k, v) in weights {
            out[k] = v.ndim == 4 ? v.transposed(0, 2, 3, 1) : v
        }
        return out
    }
}
