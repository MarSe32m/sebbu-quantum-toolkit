// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "sebbu-quantum-toolkit",
    platforms: [
        .macOS(.v26),
        .iOS(.v26),
        .tvOS(.v26),
        .watchOS(.v26)
    ],
    products: [
        .library(
            name: "SebbuQuantumToolkit",
            targets: ["SebbuQuantumToolkit"]
        ),
        .library(
            name: "SebbuQuantumToolkitGPU",
            targets: ["SebbuQuantumToolkitGPU"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/MarSe32m/sebbu-science", from: "0.4.4"),
        .package(url: "https://github.com/MarSe32m/sebbu-cuda", branch: "0.0.1"),
        .package(url: "https://github.com/apple/swift-numerics", from: "1.1.1"),
        .package(url: "https://github.com/apple/swift-collections", from: "1.6.0")
    ],
    targets: [
        .target(
            name: "SebbuQuantumToolkit",
            dependencies: [
                .product(name: "SebbuScience", package: "sebbu-science"),
                .product(name: "Numerics", package: "swift-numerics"),
                .product(name: "BasicContainers", package: "swift-collections")
            ],
            cSettings: [
                .define("ACCELERATE_NEW_LAPACK", .when(platforms: [.macOS])),
                .define("ACCELERATE_LAPACK_ILP64", .when(platforms: [.macOS]))
            ],
            swiftSettings: [
                .enableExperimentalFeature("SuppressedAssociatedTypes"),
                .enableExperimentalFeature("Lifetimes")
            ],
            linkerSettings: [
                .linkedFramework("Accelerate", .when(platforms: [.macOS]))
            ]
        ),
        .target(
            name: "SebbuQuantumToolkitGPU",
            dependencies: [
                "SebbuQuantumToolkit",
                .product(name: "SebbuScience", package: "sebbu-science"),
                .product(name: "Numerics", package: "swift-numerics"),
                //TODO: Use traits so that this is included only if trait, say "CUDA" is enabled
                .product(name: "SebbuCUDA", package: "sebbu-cuda", condition: .when(platforms: [.windows, .linux]))
            ],
            cSettings: [
                .define("ACCELERATE_NEW_LAPACK", .when(platforms: [.macOS])),
                .define("ACCELERATE_LAPACK_ILP64", .when(platforms: [.macOS]))
            ],
            linkerSettings: [
                .linkedFramework("Accelerate", .when(platforms: [.macOS]))
            ]
        ),
        .testTarget(
            name: "SebbuQuantumToolkitTests",
            dependencies: ["SebbuQuantumToolkit", "SebbuQuantumToolkitGPU"],
            cSettings: [
                .define("ACCELERATE_NEW_LAPACK", .when(platforms: [.macOS])),
                .define("ACCELERATE_LAPACK_ILP64", .when(platforms: [.macOS]))
            ],
            linkerSettings: [
                .linkedFramework("Accelerate", .when(platforms: [.macOS]))
            ]
        ),
    ],
    swiftLanguageModes: [.v6]
)
