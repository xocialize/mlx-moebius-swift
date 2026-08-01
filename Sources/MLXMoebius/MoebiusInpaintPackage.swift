import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXNN
import MLXToolKit
import MoebiusMLX
import UniformTypeIdentifiers

/// The conformant `imageInpaint` ModelPackage for Moebius — the DIFFUSION fill tier, registered
/// under its OWN PackageID (`moebius-inpaint`) beside the LaMa/MI-GAN package, per the Lucida
/// precedent: separate manifest keeps licence, provenance and footprint separable, so this tier's
/// heavier envelope can never gate LaMa's consumers. Same capability; the engine's multi-package
/// selection does the rest.
///
/// Behavior notes (each mirrors a measured or reference-verified decision):
/// - Input is resized to the model's HARD 512×512 (rel_pos_emb is spatially baked); the fill is
///   resized back and pasted through a blurred mask, so everything OUTSIDE the mask keeps original
///   resolution. Fill detail is bounded by the 512 bottleneck — inherent to the checkpoint.
/// - The VAE encode uses the posterior MEAN, not `.sample()`: deterministic output for a given
///   (image, mask, seed), the Forge preference. The reference samples; the difference is buried
///   under t=900 noising and is invisible in practice.
/// - `metaData`: `seed` (int, default 0) · `cfgScale` (double, default 2.5, the reference CLI
///   default) · `paste` (bool, default true).
@InferenceActor
public final class MoebiusInpaintPackage: ModelPackage {
    public typealias Configuration = MoebiusConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7 weights: MIT (hustvl HF card; VAE MIT via PixelHacker/SDXL). C8 port code: MIT.
            license: LicenseDeclaration(weightLicense: .mit, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "mlx-community/Moebius-Places2-fp16",
                                   revision: "main", tier: 3),
            requirements: RequirementsManifest(
                // Split footprint — IN-APP phys_footprint MEASURED 2026-08-01 (Moebius Demo,
                // ValidationHarness, isolate:true so co-residents cannot inflate the floor):
                //   floor 1.10 GB · peak 5.21 GB · activation 4.11 GB · retained-after-run 0.67 GB
                // This CORRECTED the prior smoke-derived declaration in BOTH directions: resident
                // was UNDER-declared (0.80 → 1.10 GB, +38%) and activation OVER-declared
                // (5.00 → 4.20 GB). Declared slightly above measurement for headroom.
                //
                // ⚠️ NOTE FOR THE FLEET, because it contradicts the standing rule: the BiRefNet
                // lesson says a smoke MLX-peak UNDER-reads process phys ~2.7×. Here it did the
                // OPPOSITE — smoke MLX-peak 5.65 GB vs phys peak 5.21 GB. MLX's peak counts its
                // buffer POOL, which for an allocation-heavy transient (the 512² VAE decode) can
                // exceed the resident pages phys actually reports. So "smoke under-reads" is not
                // universal; it is workload-shaped, and the in-app measurement is the arbiter
                // either way.
                footprints: [
                    QuantFootprint(quant: .fp16, residentBytes: 1_100_000_000,
                                   peakActivationBytes: 4_200_000_000)
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0))
            ),
            surfaces: [
                InpaintContract.descriptor(
                    name: "moebius-inpaint",
                    summary: "Diffusion object removal / inpainting (Moebius 0.22B, 512-native; "
                        + "image + mask → filled image). The quality tier beside LaMa's "
                        + "feed-forward fill: stronger on large semantic holes, 19 denoise steps.",
                    modes: [])
            ])
    }

    private let configuration: Configuration
    private var pipeline: MoebiusPipeline?

    public nonisolated init(configuration: Configuration) { self.configuration = configuration }

    public func load() async throws {
        guard pipeline == nil else { return }
        // Engine-executed materialization (contract 1.24) has already run for dir-less configs;
        // reaching this guard with sources missing means no store was set, or a non-engine caller.
        guard configuration.missingWeightSources(storeRoot: configuration.modelsRootDirectory)
            .isEmpty,
            let dir = configuration.resolveWeightsDirectory()
        else { throw MoebiusPackageError.weightsNotMaterialized(configuration.variant.repo) }

        let unetWeights = try MLX.loadArrays(
            url: dir.appendingPathComponent("unet.safetensors"))
        let vaeWeights = try MLX.loadArrays(url: dir.appendingPathComponent("vae.safetensors"))
        let unet = try MoebiusUNet(unetWeights)
        let vae = AutoencoderKL()
        try vae.update(parameters: ModuleParameters.unflattened(AutoencoderKL.sanitize(vaeWeights)),
                       verify: [.all])
        eval(vae)
        var built = MoebiusPipeline(unet: unet, vae: vae)
        built.numSteps = configuration.steps
        self.pipeline = built
    }

    public func unload() async {
        pipeline = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run(), before validation.
        try Task.checkCancellation()
        guard request.capability == .imageInpaint, let req = request as? InpaintRequest else {
            throw MoebiusPackageError.unsupportedCapability(request.capability)
        }
        if pipeline == nil { try await load() }
        guard var pipe = pipeline else { throw MoebiusPackageError.notLoaded }

        let sourceCG = try Self.decode(req.image)
        let maskCG = try Self.decode(req.mask)

        if let cfg = req.metaData["cfgScale"].flatMap(Self.doubleValue) {
            pipe.guidanceScale = Float(cfg)
        }
        let seed = req.metaData["seed"].flatMap(Self.intValue) ?? 0
        let paste = req.metaData["paste"].flatMap(Self.boolValue) ?? true

        // ── preprocess at the model's native 512² ────────────────────────────────────────────
        RunProgress.report(.encode)
        let model = MoebiusImageIO.prepare(source: sourceCG, mask: maskCG)
        let latents = pipe.vae.encode(model.image).mean * pipe.vae.scalingFactor
        let maskedLatents = pipe.vae.encode(model.maskedImage).mean * pipe.vae.scalingFactor
        let noise = MoebiusImageIO.offsetNoise(like: latents, seed: UInt64(bitPattern: Int64(seed)),
                                               offset: pipe.noiseOffset)
        eval(latents, maskedLatents, noise)
        try Task.checkCancellation()   // pre-denoise seam: encode is a real chunk of work

        // ── 19 denoise steps + decode; per-step cancellation + progress ──────────────────────
        let inputs = MoebiusPipeline.Inputs(latents: latents, maskedLatents: maskedLatents,
                                            maskLatent: model.maskLatent, noise: noise)
        let decoded = try pipe.run(inputs) { step, total in
            try Task.checkCancellation()
            RunProgress.report(.denoise, step: step, totalSteps: total)
        }

        // ── composite back at the ORIGINAL resolution ────────────────────────────────────────
        try Task.checkCancellation()   // pre-postprocess seam
        RunProgress.report(.postprocess)
        let output: CGImage
        if paste {
            output = try MoebiusImageIO.pasteAtOriginal(
                fill512: decoded, source: sourceCG, mask: maskCG)
        } else {
            output = try MoebiusImageIO.toCGImage(decoded)
        }
        let png = try Self.encodePNG(output)
        return InpaintResponse(image: Image(format: .png, data: png,
                                            width: output.width, height: output.height))
    }

    // MARK: metaData unwrap (MetaValue is a bare enum; the fleet pattern is local helpers)

    private nonisolated static func doubleValue(_ v: MetaValue) -> Double? {
        if case .double(let d) = v { return d }
        if case .int(let i) = v { return Double(i) }
        return nil
    }
    private nonisolated static func intValue(_ v: MetaValue) -> Int? {
        if case .int(let i) = v { return i }
        if case .double(let d) = v { return Int(d) }
        return nil
    }
    private nonisolated static func boolValue(_ v: MetaValue) -> Bool? {
        if case .bool(let b) = v { return b }
        return nil
    }

    // MARK: image codec

    private nonisolated static func decode(_ image: Image) throws -> CGImage {
        guard let src = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw MoebiusPackageError.decodeFailed }
        return cg
    }

    private nonisolated static func encodePNG(_ cg: CGImage) throws -> Data {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            data, UTType.png.identifier as CFString, 1, nil)
        else { throw MoebiusPackageError.encodeFailed }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { throw MoebiusPackageError.encodeFailed }
        return data as Data
    }

    public enum MoebiusPackageError: Error, CustomStringConvertible {
        case unsupportedCapability(Capability)
        case weightsNotMaterialized(String)
        case notLoaded
        case decodeFailed, encodeFailed

        public var description: String {
            switch self {
            case .unsupportedCapability(let c): return "moebius-inpaint cannot serve \(c)"
            case .weightsNotMaterialized(let r):
                return "weights for \(r) are not materialized (no store root and no modelDirectory)"
            case .notLoaded: return "package not loaded"
            case .decodeFailed: return "could not decode input image"
            case .encodeFailed: return "could not encode output image"
            }
        }
    }
}

public extension MoebiusInpaintPackage {
    nonisolated static var registration: PackageRegistration { .of(MoebiusInpaintPackage.self) }
}
