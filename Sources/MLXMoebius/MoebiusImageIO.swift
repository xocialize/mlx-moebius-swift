import CoreGraphics
import Foundation
import MLX
import MLXRandom
import MoebiusMLX

/// CGImage ↔ tensor plumbing for the package. Mirrors the reference preprocessing
/// (`pipeline.py::_preprocess` / `_denoise_preprocess` / `_post_process`):
/// binarize mask → resize to 512² → binarize again → masked = image·(1−mask); latent mask by
/// NEAREST 512→64; paste back through a blurred mask at the ORIGINAL resolution.
public enum MoebiusImageIO {

    public struct Prepared {
        public let image: MLXArray        // [1,3,512,512] NCHW, [-1,1]
        public let maskedImage: MLXArray  // [1,3,512,512] NCHW
        public let maskLatent: MLXArray   // [1,1,64,64] NCHW, 1 = remove
    }

    static let side = 512

    public static func prepare(source: CGImage, mask: CGImage) -> Prepared {
        let image = rgbTensor(resize(source, width: side, height: side))  // [1,3,512,512] in [0,1]
        let scaled = image * 2 - 1
        // Reference binarizes at 255/2 BEFORE the resize and re-binarizes ≥0.5 after; with a
        // high-quality resample the composition is a ≥0.5 threshold on the resized luma.
        let maskFull = binarizedMask(resize(mask, width: side, height: side))  // [1,1,512,512]
        let masked = scaled * (1 - maskFull)
        // F.interpolate default = NEAREST: index floor(i·8) = i·8, expressed as a reshape-slice.
        let m = maskFull.reshaped([1, 1, 64, 8, 64, 8])
        let maskLatent = m[0..., 0..., 0..., 0 ..< 1, 0..., 0 ..< 1].reshaped([1, 1, 64, 64])
        return Prepared(image: scaled, maskedImage: masked, maskLatent: maskLatent)
    }

    /// The reference's two RNG draws: base noise + `noise_offset`·per-channel offset.
    public static func offsetNoise(like: MLXArray, seed: UInt64, offset: Float) -> MLXArray {
        let keys = MLXRandom.split(key: MLXRandom.key(seed))
        let noise = MLXRandom.normal(like.shape, key: keys.0)
        let channelOffset = MLXRandom.normal([like.dim(0), like.dim(1), 1, 1], key: keys.1)
        return noise + offset * channelOffset
    }

    /// Composite the 512² fill back into the FULL-RESOLUTION source through a blurred mask
    /// (`_post_process`, paste=true). Only the masked region takes resampled model output; every
    /// pixel outside it keeps original resolution.
    public static func pasteAtOriginal(fill512: MLXArray, source: CGImage,
                                       mask: CGImage) throws -> CGImage {
        let w = source.width, h = source.height
        let fillCG = try toCGImage(fill512)
        let fillFull = rgbTensor(resize(fillCG, width: w, height: h))       // [1,3,h,w] [0,1]
        let sourceFull = rgbTensor(source)
        let maskFull = binarizedMask(mask)                                   // [1,1,h,w]
        let pasted = MoebiusPipeline.paste(result: fillFull, source: sourceFull, mask: maskFull)
        eval(pasted)
        return try toCGImage(pasted)
    }

    // MARK: primitives

    static func resize(_ image: CGImage, width: Int, height: Int) -> CGImage {
        guard image.width != width || image.height != height else { return image }
        guard let ctx = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return image }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return ctx.makeImage() ?? image
    }

    /// `[1,3,H,W]` NCHW float32 in `[0,1]`.
    static func rgbTensor(_ image: CGImage) -> MLXArray {
        let w = image.width, h = image.height
        var buffer = [UInt8](repeating: 0, count: w * h * 4)
        buffer.withUnsafeMutableBytes { raw in
            guard let ctx = CGContext(
                data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        let rgba = MLXArray(buffer, [1, h, w, 4]).asType(.float32) / 255
        return rgba[0..., 0..., 0..., 0 ..< 3].transposed(0, 3, 1, 2)
    }

    /// `[1,1,H,W]`, luma ≥ 0.5 → 1 (white = remove).
    static func binarizedMask(_ image: CGImage) -> MLXArray {
        let rgb = rgbTensor(image)
        let luma = rgb.mean(axes: [1], keepDims: true)
        return (luma .>= 0.5).asType(.float32)
    }

    /// NCHW `[0,1]` → CGImage.
    public static func toCGImage(_ nchw: MLXArray) throws -> CGImage {
        let h = nchw.dim(2), w = nchw.dim(3)
        let hwc = MLX.clip(nchw[0].transposed(1, 2, 0) * 255, min: 0, max: 255).asType(.uint8)
        eval(hwc)
        let rgbBytes: [UInt8] = hwc.asArray(UInt8.self)
        var rgba = [UInt8](repeating: 255, count: w * h * 4)
        for i in 0 ..< (w * h) {
            rgba[i * 4] = rgbBytes[i * 3]
            rgba[i * 4 + 1] = rgbBytes[i * 3 + 1]
            rgba[i * 4 + 2] = rgbBytes[i * 3 + 2]
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let cg = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                               bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                               bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                               provider: provider, decode: nil, shouldInterpolate: false,
                               intent: .defaultIntent)
        else { throw MoebiusInpaintPackage.MoebiusPackageError.encodeFailed }
        return cg
    }
}
