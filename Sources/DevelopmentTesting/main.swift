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

exampleHEOMRadiativeDamping(endTime: 200)
exampleHOPSRadiativeDamping(endTime: 200)
exampleQSDRadiativeDamping(endTime: 200)
exampleGKSLRadiativeDamping(endTime: 200)
exampleMCWFRadiativeDamping(endTime: 200)

exampleHEOMResonanceFluorescenceSpectrum(A: 0.27)
exampleHOPSResonanceFluorescenceSpectrum(A: 0.27, trajectories: 16384 * 2)
exampleHEOMIBM(endTime: 700)


exampleHOPSIBM(endTime: 700)

exampleTLSVibrationalModeHOPSGKSLComparison()

exampleMCWFResonanceFluorescenceSpectrum()
exampleGKSLResonanceFluorescenceSpectrum()
exampleQSDResonanceFluorescenceSpectrum()



BCFFittingExample()
hierarchyExample()
NoiseGenerationExample()
