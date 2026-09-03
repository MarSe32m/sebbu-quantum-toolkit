import ComplexModule
import Foundation
import Testing
import SebbuQuantumToolkit

private struct EncodedUnitSystemRepresentation: Encodable {
    let convention: QuantumUnitSystem.Convention
}

private func isApproximatelyEqual<T: BinaryFloatingPoint>(
    _ lhs: T,
    _ rhs: T,
    accuracy: T
) -> Bool {
    Swift.abs(lhs - rhs) <= accuracy
}

@Suite
struct QuantumUnitSystemTests {
    @Test
    func testSIAndCODATAConstants() {
        #expect(
            isApproximatelyEqual(
                PhysicalConstants.hartreeEnergy,
                4.359_744_722_206_0e-18,
                accuracy: 1e-33
            )
        )
        #expect(
            isApproximatelyEqual(
                PhysicalConstants.atomicUnitOfTime,
                2.418_884_326_586_4e-17,
                accuracy: 1e-30
            )
        )
        #expect(
            isApproximatelyEqual(
                Energy(1, .hartree).value(in: .electronVolt),
                27.211_386_245_981,
                accuracy: 2e-12
            )
        )
    }

    @Test
    func testQuantityConversionsAndArithmetic() {
        #expect(
            isApproximatelyEqual(
                Energy(2.5, .electronVolt).value(in: .milliElectronVolt),
                2_500,
                accuracy: 1e-12
            )
        )
        #expect(
            isApproximatelyEqual(
                Time(1, .picosecond).value(in: .femtosecond),
                1_000,
                accuracy: 1e-12
            )
        )
        #expect(
            isApproximatelyEqual(
                DecayRate(1, .perNanosecond).value(in: .gigaPerSecond),
                1,
                accuracy: 1e-15
            )
        )

        let sum = Energy(1, .electronVolt) + Energy(500, .milliElectronVolt)
        #expect(
            isApproximatelyEqual(
                sum.value(in: .electronVolt),
                1.5,
                accuracy: 1e-15
            )
        )
        #expect(
            isApproximatelyEqual(
                (2 * sum).value(in: .electronVolt),
                3,
                accuracy: 1e-15
            )
        )
    }

    @Test
    func testEveryDeclaredUnitRoundTrips() {
        for unit in EnergyUnit.allCases {
            #expect(
                isApproximatelyEqual(
                    Energy(1, unit).value(in: unit),
                    1,
                    accuracy: 1e-15
                )
            )
        }
        for unit in TimeUnit.allCases {
            #expect(
                isApproximatelyEqual(
                    Time(1, unit).value(in: unit),
                    1,
                    accuracy: 1e-15
                )
            )
        }
        for unit in AngularFrequencyUnit.allCases {
            #expect(
                isApproximatelyEqual(
                    AngularFrequency(1, unit).value(in: unit),
                    1,
                    accuracy: 1e-15
                )
            )
        }
        for unit in CyclicFrequencyUnit.allCases {
            #expect(
                isApproximatelyEqual(
                    CyclicFrequency(1, unit).value(in: unit),
                    1,
                    accuracy: 1e-15
                )
            )
        }
        for unit in DecayRateUnit.allCases {
            #expect(
                isApproximatelyEqual(
                    DecayRate(1, unit).value(in: unit),
                    1,
                    accuracy: 1e-15
                )
            )
        }
        for unit in TemperatureUnit.allCases {
            #expect(
                isApproximatelyEqual(
                    Temperature(1, unit).value(in: unit),
                    1,
                    accuracy: 1e-15
                )
            )
        }
        for unit in SpectroscopicWavenumberUnit.allCases {
            #expect(
                isApproximatelyEqual(
                    SpectroscopicWavenumber(1, unit).value(in: unit),
                    1,
                    accuracy: 1e-15
                )
            )
        }
    }

    @Test
    func testMultiplicativeScalesAgreeWithTypedConversions() throws {
        let units = try QuantumUnitSystem.natural(
            timeScale: 1,
            unit: .picosecond
        )

        for unit in EnergyUnit.allCases {
            #expect(
                isApproximatelyEqual(
                    units.hamiltonianScale(from: unit),
                    units.hamiltonianValue(of: Energy(1, unit)),
                    accuracy: 1e-15
                )
            )
        }
        for unit in TimeUnit.allCases {
            #expect(
                isApproximatelyEqual(
                    units.timeScale(from: unit),
                    units.value(of: Time(1, unit)),
                    accuracy: 1e-15
                )
            )
        }
        for unit in DecayRateUnit.allCases {
            #expect(
                isApproximatelyEqual(
                    units.decayRateScale(from: unit),
                    units.rateValue(of: DecayRate(1, unit)),
                    accuracy: 1e-15
                )
            )
        }
    }

    @Test
    func testNaturalSystemDefinedByEnergy() throws {
        let units = try QuantumUnitSystem.natural(
            reference: .energy(value: 1, unit: .electronVolt)
        )

        #expect(
            isApproximatelyEqual(
                units.hamiltonianValue(of: Energy(1, .electronVolt)),
                1,
                accuracy: 1e-15
            )
        )
        #expect(
            isApproximatelyEqual(
                units.internalTimeUnit.value(in: .femtosecond),
                0.658_211_956_950_906_6,
                accuracy: 2e-15
            )
        )
        #expect(units.reducedPlanckConstantInInternalUnits == 1)
    }

    @Test
    func testNaturalSystemDefinedByTime() throws {
        let units = try QuantumUnitSystem.natural(
            reference: .time(value: 1, unit: .picosecond)
        )

        #expect(
            isApproximatelyEqual(
                units.value(of: Time(1, .picosecond)),
                1,
                accuracy: 1e-15
            )
        )
        #expect(
            isApproximatelyEqual(
                units.internalEnergyUnit.value(in: .milliElectronVolt),
                0.658_211_956_950_906_6,
                accuracy: 2e-15
            )
        )
    }

    @Test
    func testHartreeAtomicSystem() {
        let units = QuantumUnitSystem.hartreeAtomic

        #expect(
            isApproximatelyEqual(
                units.hamiltonianValue(of: Energy(1, .hartree)),
                1,
                accuracy: 1e-15
            )
        )
        #expect(
            isApproximatelyEqual(
                units.value(of: Time(1, .atomicUnit)),
                1,
                accuracy: 1e-15
            )
        )
        #expect(
            isApproximatelyEqual(
                units.internalTimeUnit.value(in: .femtosecond),
                0.024_188_843_265_863_3,
                accuracy: 2e-16
            )
        )
    }

    @Test
    func testCyclicFrequencyAndDecayRateDifferByTwoPi() throws {
        let units = try QuantumUnitSystem.natural(
            reference: .time(value: 1, unit: .picosecond)
        )

        let cyclic = units.hamiltonianValue(
            of: CyclicFrequency(1, .terahertz)
        )
        let angular = units.hamiltonianValue(
            of: AngularFrequency(1, .radiansPerPicosecond)
        )
        let decay = units.rateValue(of: DecayRate(1, .perPicosecond))

        #expect(isApproximatelyEqual(cyclic, 2 * Double.pi, accuracy: 1e-14))
        #expect(isApproximatelyEqual(angular, 1, accuracy: 1e-15))
        #expect(isApproximatelyEqual(decay, 1, accuracy: 1e-15))
        #expect(
            isApproximatelyEqual(
                cyclic / decay,
                2 * Double.pi,
                accuracy: 1e-14
            )
        )
    }

    @Test
    func testTripletBatteryExampleInPicosecondUnits() throws {
        let units = try QuantumUnitSystem.natural(
            reference: .time(value: 1, unit: .picosecond)
        )

        #expect(
            isApproximatelyEqual(
                units.hamiltonianValue(of: Energy(2.34, .electronVolt)),
                3_555.085_828_035_984_4,
                accuracy: 2e-12
            )
        )
        #expect(
            isApproximatelyEqual(
                units.hamiltonianValue(of: Energy(0.25, .electronVolt)),
                379.816_861_969_656_5,
                accuracy: 2e-13
            )
        )
        #expect(
            isApproximatelyEqual(
                units.hamiltonianValue(of: Energy(35, .milliElectronVolt)),
                53.174_360_675_751_91,
                accuracy: 5e-14
            )
        )
        #expect(
            isApproximatelyEqual(
                units.value(of: Time(60, .femtosecond)),
                0.06,
                accuracy: 1e-16
            )
        )
        #expect(
            isApproximatelyEqual(
                units.rateValue(of: DecayRate(10, .gigaPerSecond)),
                0.01,
                accuracy: 1e-17
            )
        )
    }

    @Test
    func testEnergyTimeProductAndPulseAreaAreSystemIndependent() throws {
        let amplitude = Energy(20, .milliElectronVolt)
        let width = Time(30, .femtosecond)
        let expected = amplitude.valueInSIUnits
            * width.valueInSIUnits
            / PhysicalConstants.reducedPlanckConstant

        let systems = [
            QuantumUnitSystem.hartreeAtomic,
            try QuantumUnitSystem.natural(
                energyScale: 1,
                unit: .electronVolt
            ),
            try QuantumUnitSystem.natural(
                timeScale: 1,
                unit: .picosecond
            )
        ]

        for units in systems {
            let internalProduct = units.hamiltonianValue(of: amplitude)
                * units.value(of: width)
            #expect(
                isApproximatelyEqual(
                    internalProduct,
                    expected,
                    accuracy: 2e-15
                )
            )
        }
    }

    @Test
    func testTemperatureIsConvertedToThermalEnergy() throws {
        let units = try QuantumUnitSystem.natural(
            reference: .energy(value: 1, unit: .milliElectronVolt)
        )

        let thermalEnergy = units.thermalEnergyValue(
            at: Temperature(300, .kelvin)
        )
        #expect(
            isApproximatelyEqual(
                thermalEnergy,
                25.851_999_786_435_535,
                accuracy: 2e-14
            )
        )
        #expect(
            isApproximatelyEqual(
                units.temperature(fromInternalThermalEnergy: thermalEnergy)
                    .value(in: .kelvin),
                300,
                accuracy: 2e-13
            )
        )
    }

    @Test
    func testSpectroscopicWavenumber() throws {
        let units = try QuantumUnitSystem.natural(
            reference: .energy(value: 1, unit: .electronVolt)
        )

        let oneInverseCentimeter = units.hamiltonianValue(
            of: SpectroscopicWavenumber(1, .reciprocalCentimeter)
        )
        #expect(
            isApproximatelyEqual(
                oneInverseCentimeter,
                1.239_841_984_332_002_6e-4,
                accuracy: 2e-18
            )
        )

        let roundTrip = units.spectroscopicWavenumber(
            fromInternalHamiltonianValue: oneInverseCentimeter
        )
        #expect(
            isApproximatelyEqual(
                roundTrip.value(in: .reciprocalCentimeter),
                1,
                accuracy: 2e-15
            )
        )
    }

    @Test
    func testBathCorrelationScaling() throws {
        let energyUnits = try QuantumUnitSystem.natural(
            reference: .energy(value: 1, unit: .electronVolt)
        )
        #expect(
            isApproximatelyEqual(
                energyUnits.bathCorrelationValue(
                    of: EnergySquared(0.04, squared: .electronVolt)
                ),
                0.04,
                accuracy: 1e-16
            )
        )

        let timeUnits = try QuantumUnitSystem.natural(
            reference: .time(value: 1, unit: .picosecond)
        )
        #expect(
            isApproximatelyEqual(
                timeUnits.bathCorrelationValue(
                    of: InverseTimeSquared(0.04, perSquared: .picosecond)
                ),
                0.04,
                accuracy: 1e-16
            )
        )
    }

    @Test
    func testRoundTrips() throws {
        let units = try QuantumUnitSystem.natural(
            reference: .time(value: 2.5, unit: .femtosecond)
        )

        let time = Time(-14, .femtosecond)
        #expect(
            isApproximatelyEqual(
                units.time(fromInternalValue: units.value(of: time))
                    .value(in: .femtosecond),
                -14,
                accuracy: 2e-14
            )
        )

        let energy = Energy(-0.37, .electronVolt)
        #expect(
            isApproximatelyEqual(
                units.energy(
                    fromInternalHamiltonianValue:
                        units.hamiltonianValue(of: energy)
                ).value(in: .electronVolt),
                -0.37,
                accuracy: 1e-15
            )
        )

        let rate = DecayRate(7.2, .perNanosecond)
        #expect(
            isApproximatelyEqual(
                units.decayRate(
                    fromInternalValue: units.rateValue(of: rate)
                ).value(in: .perNanosecond),
                7.2,
                accuracy: 2e-15
            )
        )
    }

    @Test
    func testInvalidNaturalReferencesAreRejected() {
        #expect(throws: UnitSystemError.referenceMustBePositive) {
            try QuantumUnitSystem.natural(
                reference: .energy(value: 0, unit: .electronVolt)
            )
        }

        #expect(throws: UnitSystemError.referenceMustBeFinite) {
            try QuantumUnitSystem.natural(
                reference: .time(value: -.infinity, unit: .second)
            )
        }
    }

    @Test
    func testCodableRoundTripRevalidatesTheSystem() throws {
        let original = try QuantumUnitSystem.natural(
            reference: .time(value: 1, unit: .picosecond)
        )
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(
            QuantumUnitSystem.self,
            from: data
        )

        #expect(decoded == original)
        #expect(
            isApproximatelyEqual(
                decoded.secondsPerInternalTimeUnit,
                1e-12,
                accuracy: 1e-27
            )
        )
    }

    @Test
    func testDecodingRejectsAnInvalidReference() throws {
        let invalid = EncodedUnitSystemRepresentation(
            convention: .natural(
                reference: .energy(value: 0, unit: .electronVolt)
            )
        )
        let data = try JSONEncoder().encode(invalid)

        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(QuantumUnitSystem.self, from: data)
        }
    }

    @Test
    func testComplexScalingAndBathExponent() throws {
        let units = try QuantumUnitSystem.natural(
            reference: .time(value: 1, unit: .picosecond)
        )

        let coefficient = Complex<Double>(1, -0.5)
        let scaled = units.bathCorrelationValue(
            of: coefficient,
            perSquared: .picosecond
        )
        #expect(isApproximatelyEqual(scaled.real, 1, accuracy: 1e-15))
        #expect(isApproximatelyEqual(scaled.imaginary, -0.5, accuracy: 1e-15))

        let floatEnergy = Complex<Float>(1, -0.25)
        let scaledFloatEnergy = units.hamiltonianValue(
            of: floatEnergy,
            expressedIn: .milliElectronVolt
        )
        let expectedFloatScale = Float(
            units.hamiltonianScale(from: .milliElectronVolt)
        )
        #expect(scaledFloatEnergy.real == expectedFloatScale)
        #expect(
            scaledFloatEnergy.imaginary == -0.25 * expectedFloatScale
        )

        let exponent = units.bathExponentValue(
            decayRate: DecayRate(10, .gigaPerSecond),
            angularFrequency: AngularFrequency(0.65, .radiansPerPicosecond)
        )
        #expect(isApproximatelyEqual(exponent.real, 0.01, accuracy: 1e-17))
        #expect(
            isApproximatelyEqual(
                exponent.imaginary,
                0.65,
                accuracy: 1e-15
            )
        )
    }
}
