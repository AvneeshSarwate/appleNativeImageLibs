// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "CoreMLPipeline",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "CoreMLPipeline",
            path: "Sources",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
