// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import PythonKit
import SebbuPythonKit

#if os(macOS)
PythonLibrary.useLibrary(at: "/Library/Frameworks/Python.framework/Versions/3.12/Python")
#elseif os(Linux) && !canImport(Musl)
// PythonLibrary.useLibrary(at: "/usr/lib64/libpython3.11.so.1.0")
PythonLibrary.useLibrary(at: "/usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0")
#elseif os(Windows)
//TODO: Set library path on Windows machine
#endif

if CommandLine.arguments.contains("--stochastic-potential") {
    try exampleHOPSStochasticPotentials()
} else if CommandLine.arguments.contains("--thermal-pseudomode") {
    try exampleHOPSThermalPseudomodeComparison()
} else if CommandLine.arguments.contains("--ibm") {
    exampleHOPSIBM(endTime: 7)
    exampleHEOMIBM(endTime: 7)
} else if CommandLine.arguments.contains("--radiative-damping") {
    exampleHEOMRadiativeDamping(endTime: 200)
    exampleHOPSRadiativeDamping(endTime: 200)
    exampleQSDRadiativeDamping(endTime: 200)
    exampleGKSLRadiativeDamping(endTime: 200)
    exampleMCWFRadiativeDamping(endTime: 200)
} else if CommandLine.arguments.contains("--resonance-fluorescence") {
    exampleHEOMResonanceFluorescenceSpectrum(A: 0.27)
    exampleHOPSResonanceFluorescenceSpectrum(A: 0.27, trajectories: 8192)
    exampleMCWFResonanceFluorescenceSpectrum()
    exampleGKSLResonanceFluorescenceSpectrum()
    exampleQSDResonanceFluorescenceSpectrum()
} else {
    exampleTLSVibrationalModeHOPSGKSLComparison()
    BCFFittingExample()
    hierarchyExample()
    NoiseGenerationExample()
}
