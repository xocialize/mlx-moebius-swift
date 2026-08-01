import Foundation
import MLXToolKit

/// Which published Moebius checkpoint to load. Only the Places2 fine-tune ships in v0.1 —
/// it is the natural-scene checkpoint, which is the Forge Erase use case. The portrait
/// fine-tunes (CelebA-HQ / FFHQ) and the pretrained base can join as cases when published.
public enum MoebiusVariant: String, Codable, Sendable, CaseIterable {
    case places2

    /// One consolidated repo per variant: UNet + VAE + card. The VAE rides in the same repo
    /// (the IndexTTS/MOSS consolidation precedent) — it is fp16-exact upstream PixelHacker/SDXL
    /// and has no second consumer, so a separate repo would only complicate MAT.
    public var repo: String {
        switch self {
        case .places2: return "mlx-community/Moebius-Places2-fp16"
        }
    }
}

/// Init-time configuration for `MoebiusInpaintPackage` (C9).
public struct MoebiusConfiguration: PackageConfiguration, ModelStorable, QuantConfigured,
    WeightSourcing {
    public var variant: MoebiusVariant
    /// fp16 is the published dtype — MEASURED (per-forward rel 9.257e-04, at the GPU fp32 noise
    /// floor; bf16 rejected at ~9× worse). `.fp16` here is both the quant-tier label and the
    /// actual load dtype; BatchNorm running statistics inside the file are fp32 by construction.
    public var quant: Quant
    /// Denoise steps (reference default 20; `strength 0.99` makes the effective count 19).
    public var steps: Int
    /// Explicit weights directory (must contain `unet.safetensors` + `vae.safetensors`) —
    /// the escape hatch, honored before any store probe.
    public var modelDirectory: URL?
    /// Set by the engine from its `ModelStore`; excluded from `Codable`.
    public var modelsRootDirectory: URL?

    public init(variant: MoebiusVariant = .places2, quant: Quant = .fp16, steps: Int = 20,
                modelDirectory: URL? = nil, modelsRootDirectory: URL? = nil) {
        self.variant = variant
        self.quant = quant
        self.steps = steps
        self.modelDirectory = modelDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey { case variant, quant, steps }

    // MARK: WeightSourcing

    /// Both files declared explicitly: a half-materialized snapshot (UNet without VAE) must read
    /// as missing rather than fail at load — the strict-probe lesson from TRELLIS.
    public var weightSources: [WeightSource] {
        [WeightSource(role: "main", repo: variant.repo,
                      matching: ["unet.safetensors", "vae.safetensors"])]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if let dir = modelDirectory {
            let haveBoth = FileManager.default.fileExists(
                atPath: dir.appendingPathComponent("unet.safetensors").path)
                && FileManager.default.fileExists(
                    atPath: dir.appendingPathComponent("vae.safetensors").path)
            if haveBoth { return [] }
        }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }

    /// Directory the loader reads from: escape hatch, then the engine's flat store layout, then a
    /// hub-client snapshot. `nil` means materialization has not happened (load()'s offline guard).
    public func resolveWeightsDirectory() -> URL? {
        if let dir = modelDirectory { return dir }
        guard let root = modelsRootDirectory else { return nil }
        let store = ModelStore(root: root)
        if let flat = store.directory(for: variant.repo),
           FileManager.default.fileExists(atPath: flat.appendingPathComponent("unet.safetensors").path) {
            return flat
        }
        if let snapshot = store.snapshotDirectory(for: variant.repo, revision: nil),
           FileManager.default.fileExists(atPath: snapshot.appendingPathComponent("unet.safetensors").path) {
            return snapshot
        }
        return nil
    }
}
