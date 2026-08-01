// swift-tools-version: 6.2
// mlx-moebius-swift — Moebius (hustvl) image inpainting on Swift/MLX.
//
// A 0.22B latent-diffusion inpainter distilled from PixelHacker, Apache-2.0 code / MIT weights.
// Destined for MLXEngine as a SECOND `imageInpaint` provider beside LaMa, under its own PackageID
// (`moebius-inpaint`) — per the Lucida precedent, a separate manifest keeps licence, provenance and
// footprint separable so the diffusion tier never gates LaMa's consumers.
//
// TWO products, the fleet convention:
//   • MoebiusMLX  — engine-agnostic Swift/MLX core (mlx-swift only; usable standalone)
//   • MLXMoebius  — the MLXEngine `imageInpaint` ModelPackage over that core   [Stage 2, pending]
//
// Gates run as EXECUTABLES, not XCTests: the SPM test host cannot resolve the mlx-swift metallib,
// while `swift run` does real GPU inference. Build with `--build-system swiftbuild`.
import PackageDescription

let package = Package(
    name: "mlx-moebius-swift",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "MoebiusMLX", targets: ["MoebiusMLX"]),
        .executable(name: "moebius-gate", targets: ["MoebiusGate"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.3"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "MoebiusMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "MoebiusGate",
            dependencies: [
                "MoebiusMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
