import Foundation
import MLX

/// LλMI — Local-λ Mix Interaction, Moebius' linear-attention block.
///
/// This is **LambdaNetworks** (Bello, ICLR 2021) as adapted in `model_lib/nets/layers/λ/vanillaλ.py`,
/// itself derived from lucidrains' MIT implementation. Structure is kept isomorphic to that file:
/// same class names, same forward order, same einsum index letters in the comments — so the Swift
/// and PyTorch sources can be read side by side.
///
/// Why it is linear: K and V are projected to a FIXED small width (`dimK` = 40 at every level,
/// shared across all heads) and contracted with each other FIRST, producing a `[dimK, dimV]` lambda
/// matrix. An `[N, N]` attention matrix is never formed — at the 64×64 level that would be
/// 4096², which is the whole point of the architecture.
///
/// Two branches, and Moebius uses one of each:
///   • self-attention  → LOCAL  (`posConv`, a 15×15 depthwise conv over V)
///   • cross-attention → GLOBAL (`relPosEmb`, a learned [N, M, dimK] table; M = 10 category tokens)
public enum Lambda {

    /// BatchNorm in INFERENCE form, written out explicitly rather than as an `MLXNN.BatchNorm`.
    ///
    /// ⚠️ DELIBERATE, and load-bearing. Moebius carries **124** BatchNorm modules
    /// (`bn1`/`bn2`/`norm_q`/`norm_v`). `MLXNN.Module.training` defaults to TRUE, and in that state
    /// BatchNorm normalises by the CURRENT batch's statistics and overwrites the checkpoint's
    /// running stats on every forward — output still looks like an image, so eyeball review misses
    /// it (PROD BiRefNet was 68% off this way; see memory `mlx-swift-eval-mode-c14`).
    /// Implementing the inference math directly means there is no `training` flag to forget, and no
    /// choke point that a future call site can bypass. Engine C14 is satisfied by construction.
    @inlinable
    public static func batchNormInference(
        _ x: MLXArray, weight: MLXArray, bias: MLXArray,
        runningMean: MLXArray, runningVar: MLXArray, eps: Float = 1e-5
    ) -> MLXArray {
        // Channel axis is last (NHWC / [B, L, C]); broadcasting handles the rest.
        let normalized = (x - runningMean) * rsqrt(runningVar + eps)
        return normalized * weight + bias
    }

    /// The `b h k n, b k v n -> b h v n` (local) and `b h k n, b n k v -> b h v n` (global)
    /// contraction, shared by both branches.
    ///
    /// Both contract over `k` with `n` acting as a BATCH axis, which is not a plain matmul. Written
    /// as a broadcast product it would materialise `b·h·k·v·n` — 52M floats at the top level — so
    /// `n` is moved into the batch position and it becomes a batched `[h,k] @ [k,v]` instead.
    ///
    /// - Parameters:
    ///   - Q: `[b, h, k, n]`
    ///   - lambdaP: `[b, n, k, v]`
    /// - Returns: `[b, h, v, n]`
    @inlinable
    public static func applyPositionalLambda(_ Q: MLXArray, _ lambdaP: MLXArray) -> MLXArray {
        let qbn = Q.transposed(0, 3, 1, 2)      // [b, n, h, k]
        let y = matmul(qbn, lambdaP)            // [b, n, h, v]
        return y.transposed(0, 2, 3, 1)         // [b, h, v, n]
    }
}

/// Self-attention branch — LOCAL context (`r = 15`, so `posConv` is used and `relPosEmb` is absent).
public struct MultiQuerySelfLambda {
    public let toQ: MLXArray          // [dimK*heads, dim, 1, 1] — 1×1 conv, no bias
    public let toK: MLXArray          // [dimK*dimU,  dim, 1, 1]
    public let toV: MLXArray          // [dimV*dimU,  dim, 1, 1]
    public let normQ: BatchNormParams
    public let normV: BatchNormParams
    public let posConvWeight: MLXArray // [dimK, dimU, 1, r, r]
    public let posConvBias: MLXArray
    public let heads: Int
    public let dimU: Int
    public let r: Int

    public init(toQ: MLXArray, toK: MLXArray, toV: MLXArray,
                normQ: BatchNormParams, normV: BatchNormParams,
                posConvWeight: MLXArray, posConvBias: MLXArray,
                heads: Int = 8, dimU: Int = 1, r: Int = 15) {
        self.toQ = toQ; self.toK = toK; self.toV = toV
        self.normQ = normQ; self.normV = normV
        self.posConvWeight = posConvWeight; self.posConvBias = posConvBias
        self.heads = heads; self.dimU = dimU; self.r = r
    }

    /// - Parameter x: `[b, hh, ww, c]` (NHWC). Returns `[b, hh, ww, heads*dimV]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), hh = x.dim(1), ww = x.dim(2)
        let n = hh * ww
        let h = heads, u = dimU

        // A 1×1 conv is a channel matmul. `squeezed` drops the two trailing kernel axes.
        let q = matmul(x, toQ.squeezed(axes: [2, 3]).transposed(1, 0))
        let k0 = matmul(x, toK.squeezed(axes: [2, 3]).transposed(1, 0))
        let v0 = matmul(x, toV.squeezed(axes: [2, 3]).transposed(1, 0))

        let Qn = Lambda.batchNormInference(q, weight: normQ.weight, bias: normQ.bias,
                                           runningMean: normQ.runningMean,
                                           runningVar: normQ.runningVar)
        let Vn = Lambda.batchNormInference(v0, weight: normV.weight, bias: normV.bias,
                                           runningMean: normV.runningMean,
                                           runningVar: normV.runningVar)

        let dimK = Qn.shape.last! / h
        let dimV = Vn.shape.last! / u

        // 'b (h k) hh ww -> b h k (hh ww)', reached from NHWC.
        let Q = Qn.reshaped([b, n, h, dimK]).transposed(0, 2, 3, 1)     // [b,h,k,n]
        var K = k0.reshaped([b, n, u, dimK]).transposed(0, 2, 3, 1)     // [b,u,k,m]
        let V = Vn.reshaped([b, n, u, dimV]).transposed(0, 2, 3, 1)     // [b,u,v,m]

        // Content term. softmax runs over the CONTEXT axis (m), never over n×n.
        K = softmax(K, axis: -1)
        let lambdaC = matmul(K, V.transposed(0, 1, 3, 2)).sum(axis: 1)  // 'b u k m, b u v m -> b k v'
        let Yc = matmul(lambdaC.transposed(0, 2, 1).expandedDimensions(axis: 1), Q)

        // Positional term (LOCAL). The reference applies Conv3d(dimU -> dimK, (1,r,r)) to V viewed
        // as [b, u, v, hh, ww]. With dimU == 1 and kernel depth 1 that is exactly a 2-D conv run
        // independently over each of the v depth-slices, so v folds into the batch axis.
        //
        // ⚠️ NOT expressed as conv2d — measured 2026-08-01: MLX's conv path for a single-input-
        // channel 15×15 kernel runs at <1% of GEMM efficiency (20.8 ms at 64², which made this
        // ONE op ~half the whole UNet forward). The identical computation as an im2col unfold +
        // one [·, r·r] @ [r·r, dimK] matmul is 2.3 ms (9×), rel 6e-04 in fp16 — pure reduction
        // order. Revisit if MLX ever ships a fast small-channel conv.
        var Vp = Vn.reshaped([b, n, u, dimV])
        Vp = Vp.transposed(0, 2, 3, 1).reshaped([b * u * dimV, hh, ww])
        let pad = r / 2
        let sideH = hh + 2 * pad, sideW = ww + 2 * pad
        let xp = padded(Vp, widths: [IntOrPair(0), IntOrPair(pad), IntOrPair(pad)])
        let col = asStrided(xp, [b * u * dimV, hh, ww, r, r],
                            strides: [sideH * sideW, sideW, 1, sideW, 1])
        let wG = posConvWeight.squeezed(axes: [1, 2]).reshaped([dimK, r * r]).transposed(1, 0)
        var lambdaP = matmul(col.reshaped([b * u * dimV, n, r * r]), wG) + posConvBias
        lambdaP = lambdaP.reshaped([b, u * dimV, n, dimK]).transposed(0, 2, 3, 1)  // [b,n,k,v]
        let Yp = Lambda.applyPositionalLambda(Q, lambdaP)

        let Y = Yc + Yp                                                  // [b,h,v,n]
        return Y.transposed(0, 3, 1, 2).reshaped([b, hh, ww, h * dimV])
    }
}

/// Cross-attention branch — GLOBAL context (`relPosEmb`; M = 10 category-embedding tokens).
public struct MultiQueryCrossLambda {
    public let toQ: MLXArray          // [dimK*heads, dim, 1, 1] — 1×1 conv
    public let toK: MLXArray          // [dimK*dimU, dimCross]   — LINEAR
    public let toV: MLXArray          // [dimV*dimU, dimCross]   — LINEAR
    public let normQ: BatchNormParams
    public let normV: BatchNormParams
    public let relPosEmb: MLXArray    // [n*n, m, dimK, dimU]
    public let heads: Int
    public let dimU: Int

    public init(toQ: MLXArray, toK: MLXArray, toV: MLXArray,
                normQ: BatchNormParams, normV: BatchNormParams,
                relPosEmb: MLXArray, heads: Int = 8, dimU: Int = 1) {
        self.toQ = toQ; self.toK = toK; self.toV = toV
        self.normQ = normQ; self.normV = normV
        self.relPosEmb = relPosEmb; self.heads = heads; self.dimU = dimU
    }

    /// - Parameters:
    ///   - x: `[b, hh, ww, c]`
    ///   - hiddenStates: `[b, m, dimCross]` — the 10 category embeddings, projected to 768.
    public func callAsFunction(_ x: MLXArray, hiddenStates: MLXArray) -> MLXArray {
        let b = x.dim(0), hh = x.dim(1), ww = x.dim(2)
        let n = hh * ww
        let m = hiddenStates.dim(1)
        let h = heads, u = dimU

        let q = matmul(x, toQ.squeezed(axes: [2, 3]).transposed(1, 0))
        let k0 = matmul(hiddenStates, toK.transposed(1, 0))
        let v0 = matmul(hiddenStates, toV.transposed(1, 0))

        let Qn = Lambda.batchNormInference(q, weight: normQ.weight, bias: normQ.bias,
                                           runningMean: normQ.runningMean,
                                           runningVar: normQ.runningVar)
        // norm_v is BatchNorm1d over the reference's 'b c l' arrangement — i.e. it normalises the
        // CHANNEL axis, which in our [b, m, c] layout is still the last one.
        let Vn = Lambda.batchNormInference(v0, weight: normV.weight, bias: normV.bias,
                                           runningMean: normV.runningMean,
                                           runningVar: normV.runningVar)

        let dimK = Qn.shape.last! / h
        let dimV = Vn.shape.last! / u

        let Q = Qn.reshaped([b, n, h, dimK]).transposed(0, 2, 3, 1)      // [b,h,k,n]
        var K = k0.reshaped([b, m, u, dimK]).transposed(0, 2, 3, 1)      // [b,u,k,l]
        let V = Vn.reshaped([b, m, u, dimV]).transposed(0, 2, 3, 1)      // [b,u,v,l]

        K = softmax(K, axis: -1)
        let lambdaC = matmul(K, V.transposed(0, 1, 3, 2)).sum(axis: 1)   // [b,k,v]
        let Yc = matmul(lambdaC.transposed(0, 2, 1).expandedDimensions(axis: 1), Q)

        // Positional term (GLOBAL). The reference indexes `rel_pos_emb[n, m]` with an
        // arange × arange meshgrid — an IDENTITY gather — so the parameter is used as-is;
        // reproducing the gather would only cost memory.
        // 'n m k u, b u v m -> b n k v', with dimU == 1.
        let rel = relPosEmb.squeezed(axis: 3)                            // [n, m, k]
        let relNK = rel.transposed(0, 2, 1).reshaped([n * dimK, m])      // [n*k, m]
        let Vm = V.squeezed(axis: 1)                                     // [b, v, m]
        let lambdaP = matmul(relNK, Vm.transposed(0, 2, 1))              // [b, n*k, v]
            .reshaped([b, n, dimK, dimV])
        let Yp = Lambda.applyPositionalLambda(Q, lambdaP)

        let Y = Yc + Yp
        return Y.transposed(0, 3, 1, 2).reshaped([b, hh, ww, h * dimV])
    }
}

/// The four tensors a BatchNorm needs in inference form.
public struct BatchNormParams {
    public let weight: MLXArray
    public let bias: MLXArray
    public let runningMean: MLXArray
    public let runningVar: MLXArray

    public init(weight: MLXArray, bias: MLXArray, runningMean: MLXArray, runningVar: MLXArray) {
        self.weight = weight; self.bias = bias
        self.runningMean = runningMean; self.runningVar = runningVar
    }

    /// Pull `<prefix>.{weight,bias,running_mean,running_var}` out of a flat weight dictionary.
    public init(_ weights: [String: MLXArray], _ prefix: String) throws {
        func need(_ suffix: String) throws -> MLXArray {
            guard let v = weights["\(prefix).\(suffix)"] else {
                throw MoebiusError.missingWeight("\(prefix).\(suffix)")
            }
            return v
        }
        self.init(weight: try need("weight"), bias: try need("bias"),
                  runningMean: try need("running_mean"), runningVar: try need("running_var"))
    }
}

public enum MoebiusError: Error, CustomStringConvertible {
    case missingWeight(String)
    case fixtureNotFound(String)

    public var description: String {
        switch self {
        case .missingWeight(let k): return "missing weight: \(k)"
        case .fixtureNotFound(let p): return "fixture not found: \(p)"
        }
    }
}
