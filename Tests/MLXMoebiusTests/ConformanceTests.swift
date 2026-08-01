// ConformanceTests.swift — MoebiusInpaintPackage through the engine's offline gates.
// Nothing here runs an MLX kernel: CAN-1/2 throw at the entry checkpoint before any weight is
// touched, MAT is filesystem-only, and the manifest checks are pure values. Run with
// `swift test --build-system swiftbuild` (the `native` build system ships no metallib and aborts
// the whole xctest process — memory `mlx-swift-metallib-ci-build-system`).

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXMoebius

final class ConformanceTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        let package = MoebiusInpaintPackage(configuration: MoebiusConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: InpaintRequest(image: Image(format: .png, data: Data()),
                                    mask: Image(format: .png, data: Data())))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // imageInpaint is not in longRunCapabilities, but this package's 3.5 GB declared peak
        // and 19-step loop make the honest posture a cadence, not the sub-second exemption:
        // per-DDIM-step throwing onStep in MoebiusPipeline.run, bracketed by post-encode and
        // pre-postprocess seams (single evals — seams, not recurring units).
        let report = CancellationConformance.checkCadence(
            manifest: MoebiusInpaintPackage.manifest,
            posture: .cadence([
                .init(phase: .denoise, unit: .step)
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - MAT-1..5 — WeightSourcing declarations

    func testMATGate() throws {
        // Satisfied = the modelDirectory escape hatch pointing at a dir holding both files —
        // the same both-files-or-missing rule the store probe enforces (a half-materialized
        // snapshot must read missing rather than fail at load).
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("moebius-mat-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data("x".utf8).write(to: dir.appendingPathComponent("unet.safetensors"))
        try Data("x".utf8).write(to: dir.appendingPathComponent("vae.safetensors"))

        let report = MaterializationConformance.check(
            freshConfiguration: MoebiusConfiguration(),
            satisfiedConfiguration: MoebiusConfiguration(modelDirectory: dir))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testHalfMaterializedDirectoryReadsMissing() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("moebius-half-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data("x".utf8).write(to: dir.appendingPathComponent("unet.safetensors"))
        // vae.safetensors deliberately absent → the source must still report missing.
        let config = MoebiusConfiguration(modelDirectory: dir)
        XCTAssertFalse(config.missingWeightSources(storeRoot: nil).isEmpty,
                       "a directory holding only the UNet must not read as satisfied")
    }

    // MARK: - manifest statics (C-level spot checks)

    func testManifestLicenseAndSurfaces() {
        let m = MoebiusInpaintPackage.manifest
        XCTAssertEqual(m.license.weightLicense, .mit)          // C7
        XCTAssertEqual(m.license.portCodeLicense, .mit)        // C8
        XCTAssertEqual(m.surfaces.count, 1)
        XCTAssertEqual(m.surfaces[0].capability, .imageInpaint)
        XCTAssertEqual(m.surfaces[0].name, "moebius-inpaint")  // own PackageID surface
        XCTAssertTrue(m.surfaces[0].supportedModes.isEmpty)    // single diffusion tier, no modes
        XCTAssertEqual(m.requirements.footprints.count, 1)     // fp16 only — the measured verdict
        XCTAssertEqual(m.requirements.footprints[0].quant, .fp16)
        XCTAssertGreaterThan(m.requirements.footprints[0].peakActivationBytes ?? 0, 0,
                             "split footprint: activation must be declared, not folded into resident")
    }

    func testConfigurationCodableRoundTrip() throws {
        var config = MoebiusConfiguration(variant: .places2, quant: .fp16, steps: 20)
        config.modelsRootDirectory = URL(fileURLWithPath: "/tmp")   // must NOT survive encoding
        let data = try JSONEncoder().encode(config)
        let back = try JSONDecoder().decode(MoebiusConfiguration.self, from: data)
        XCTAssertEqual(back.variant, .places2)
        XCTAssertEqual(back.quant, .fp16)
        XCTAssertEqual(back.steps, 20)
        XCTAssertNil(back.modelsRootDirectory,
                     "store root is engine-injected state, never serialized configuration")
    }
}
