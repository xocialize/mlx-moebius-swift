import ArgumentParser
import Foundation
import MLX
import MLXMoebius
import MLXServeCore
import MLXToolKit

/// The LIVE engine gate: register → (cold) engine-executed materialization → prepare → run →
/// evict, through a real `MLXServeEngine` against the PUBLISHED weights.
///
/// With no downloader left in the package, `prepare()` succeeding on a THROWAWAY store IS the
/// engine-executor live proof (the gepard/z-image pattern — a bench instead of a test). Run with
/// a temp store for the cold `[MAT]` figure; point at the shared store for a warm validation run.
extension MoebiusGate {
    func runEngineGate(store storePath: String?, image: String, mask: String,
                       output: String?) async throws {
        let storeRoot: URL
        var throwaway = false
        if let storePath {
            storeRoot = URL(fileURLWithPath: storePath)
        } else {
            storeRoot = FileManager.default.temporaryDirectory
                .appendingPathComponent("moebius-engine-gate-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: storeRoot, withIntermediateDirectories: true)
            throwaway = true
        }
        defer { if throwaway { try? FileManager.default.removeItem(at: storeRoot) } }
        print("[ENG] store=\(storeRoot.path) throwaway=\(throwaway)")

        let engine = MLXServeEngine()          // default policy — no acknowledgement flow (MIT/MIT)
        await engine.useModelStore(ModelStore(root: storeRoot))

        let id = try await engine.register(MoebiusInpaintPackage.registration,
                                           configuration: MoebiusConfiguration())
        let cold = await engine.needsDownload(.imageInpaint, package: id)
        print("[ENG] registered id=\(id.rawValue) needsDownload=\(cold)")

        let t0 = Date()
        _ = try await engine.prepare(.imageInpaint, package: id)
        let prepSecs = Date().timeIntervalSince(t0)
        print(String(format: "[MAT] prepare %.1f s (cold download+load = engine-executor live proof: "
                     + "the package has no downloader)", prepSecs))

        let imageData = try Data(contentsOf: URL(fileURLWithPath: image))
        let maskData = try Data(contentsOf: URL(fileURLWithPath: mask))
        let request = InpaintRequest(image: Image(format: .png, data: imageData),
                                     mask: Image(format: .png, data: maskData))
        MLX.GPU.resetPeakMemory()
        let t1 = Date()
        let response = try await engine.run(request, package: id)
        let runSecs = Date().timeIntervalSince(t1)
        guard let inpaint = response as? InpaintResponse else { throw ExitCode(1) }
        print(String(format: "[RUN] %.1f s  out %dx%d  MLX peak %.2f GB",
                     runSecs, inpaint.image.width ?? 0, inpaint.image.height ?? 0,
                     Double(MLX.Memory.peakMemory) / 1e9))
        if let output {
            try inpaint.image.data.write(to: URL(fileURLWithPath: output))
            print("[RUN] wrote \(output)")
        }

        await engine.evict(package: id)
        print("[ENG] evicted — ENGINE GATE: PASS")
    }
}
