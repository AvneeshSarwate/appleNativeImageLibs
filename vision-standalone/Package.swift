// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VisionStandalone",
    platforms: [.macOS(.v12)],
    targets: [
        .executableTarget(
            name: "VisionApp",
            path: "Sources/VisionApp",
            swiftSettings: [
                .swiftLanguageMode(.v5),
                .unsafeFlags(["-F", "../Syphon-Framework/build/Release"]),
            ],
            linkerSettings: [
                .unsafeFlags([
                    "-F", "../Syphon-Framework/build/Release",
                    "-framework", "Syphon",
                    "-Xlinker", "-rpath", "-Xlinker",
                    "@loader_path/Frameworks"
                ])
            ]
        ),
    ]
)
