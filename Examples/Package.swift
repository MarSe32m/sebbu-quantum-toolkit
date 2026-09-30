// swift-tools-version: 6.4

import PackageDescription
import Foundation

let useLocalSQT = Bool(ProcessInfo.processInfo.environment["USE_LOCAL_SQT"] ?? "false") ?? false

let SQTDependency = if useLocalSQT {
    Package.Dependency.package(name: "sebbu-quantum-toolkit", path: "..")
} else {
    Package.Dependency.package(url: "https://github.com/MarSe32m/sebbu-quantum-toolkit", branch: "main")
}

let package = Package(
    name: "Examples",
    platforms: [
        .macOS(.v27),
        .iOS(.v27),
        .tvOS(.v27),
        .watchOS(.v27)
    ],
    dependencies: [
        SQTDependency,
        .package(url: "https://github.com/MarSe32m/sebbu-science", from: "0.4.15"),
        .package(url: "https://github.com/MarSe32m/sebbu-python-kit", from: "0.0.1"),
        .package(url: "https://github.com/apple/swift-numerics", from: "1.1.1")
    ],
    targets: [
        .executableTarget(
            name: "Examples",
            dependencies: [
                .product(name: "SebbuQuantumToolkit", package: "sebbu-quantum-toolkit"),
                .product(name: "SebbuQuantumToolkitGPU", package: "sebbu-quantum-toolkit"),
                .product(name: "SebbuScience", package: "sebbu-science"),
                .product(name: "SebbuPythonKit", package: "sebbu-python-kit"),
                .product(name: "Numerics", package: "swift-numerics")
            ],
            cSettings: [
                .define("ACCELERATE_NEW_LAPACK", .when(platforms: [.macOS])),
                .define("ACCELERATE_LAPACK_ILP64", .when(platforms: [.macOS]))
            ],
            swiftSettings: [
                .unsafeFlags(["-cross-module-optimization"], .when(configuration: .release))
            ],
            linkerSettings: [
                .linkedFramework("Accelerate", .when(platforms: [.macOS]))
            ]
        ),
    ],
    swiftLanguageModes: [.v6]
)
