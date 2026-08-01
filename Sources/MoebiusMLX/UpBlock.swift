import Foundation
import MLX

/// diffusers `Upsample2D` — nearest ×2 followed by a dense 3×3 conv.
public struct Upsample2D {
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
        // Nearest ×2: each row/column duplicated in place (np.repeat semantics), NOT tiled — the
        // tile-vs-repeat confusion is the classic source of stride-2 checkerboard artefacts.
        var t = repeated(x, count: 2, axis: 1)
        t = repeated(t, count: 2, axis: 2)
        return conv2d(t, weight, padding: IntOrPair(1)) + bias
    }
}

/// `DWMixTFUpBlock2D` — `[concat skip → resnet → attention] × numLayers`, then an optional upsample.
///
/// ⚠️ Skips are consumed **LIFO**: the reference pops `res_hidden_states_tuple[-1]` each iteration
/// and shortens the tuple from the end. Feeding them in encounter order silently pairs every
/// resnet with the wrong resolution's features — and since the channel counts often still line up
/// (here 1280, 1280, then 640), it can run to completion producing plausible garbage.
public struct DWMixTFUpBlock2D {
    public let resnets: [DWResnetBlock2D]
    public let attentions: [MixTransformer2DModel]
    public let upsampler: Upsample2D?

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
        self.upsampler = try? Upsample2D(weights, prefix: "\(p)upsamplers.0")
    }

    /// - Parameter skips: in the order the down path PRODUCED them; consumed from the end.
    public func callAsFunction(_ x: MLXArray, skips: [MLXArray], temb: MLXArray,
                               context: MLXArray) -> MLXArray {
        var h = x
        var remaining = skips
        for (resnet, attn) in zip(resnets, attentions) {
            let skip = remaining.removeLast()                  // LIFO
            h = concatenated([h, skip], axis: -1)              // channel concat (axis 1 in NCHW)
            h = resnet(h, temb: temb)
            h = attn(h, context: context)
            eval(h)
        }
        if let upsampler {
            h = upsampler(h)
            eval(h)
        }
        return h
    }
}
