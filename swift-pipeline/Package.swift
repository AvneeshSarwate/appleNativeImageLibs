// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "CoreMLPipeline",
    platforms: [.macOS(.v14)],
    targets: [
        .target(
            name: "Shared",
            path: "Sources/Shared",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "CoreMLPipeline",
            path: "Sources/Benchmark",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "CameraApp",
            dependencies: ["Shared"],
            path: "Sources/CameraApp",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
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
                    "@loader_path/../../../../Syphon-Framework/build/Release"
                ])
            ]
        ),
        .target(
            name: "HandPoseInvestigation",
            path: "Sources/HandPoseInvestigation",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "HandPoseReplay",
            dependencies: ["HandPoseInvestigation"],
            path: "Sources/HandPoseReplay",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "HandPoseMatrix",
            dependencies: ["HandPoseInvestigation"],
            path: "Sources/HandPoseMatrix",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "SyphonVideoPublisher",
            dependencies: ["HandPoseInvestigation"],
            path: "Sources/SyphonVideoPublisher",
            swiftSettings: [
                .swiftLanguageMode(.v5),
                .unsafeFlags(["-F", "../Syphon-Framework/build/Release"]),
            ],
            linkerSettings: [
                .unsafeFlags([
                    "-F", "../Syphon-Framework/build/Release",
                    "-framework", "Syphon",
                    "-Xlinker", "-rpath", "-Xlinker",
                    "@loader_path/../../../../Syphon-Framework/build/Release"
                ])
            ]
        ),
        .executableTarget(
            name: "SyphonHandPoseProbe",
            dependencies: ["HandPoseInvestigation"],
            path: "Sources/SyphonHandPoseProbe",
            swiftSettings: [
                .swiftLanguageMode(.v5),
                .unsafeFlags(["-F", "../Syphon-Framework/build/Release"]),
            ],
            linkerSettings: [
                .unsafeFlags([
                    "-F", "../Syphon-Framework/build/Release",
                    "-framework", "Syphon",
                    "-Xlinker", "-rpath", "-Xlinker",
                    "@loader_path/../../../../Syphon-Framework/build/Release"
                ])
            ]
        ),
    ]
)
