// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

@usableFromInline
internal struct OutputCursor {
    @usableFromInline
    let timeSpan: SimulationTimeSpan
    
    @usableFromInline
    let schedule: OutputSchedule
    
    @usableFromInline
    var currentTime: Double
    
    @usableFromInline
    var currentTimeIndex: Int = 0
    
    @inlinable
    public init(timeSpan: SimulationTimeSpan, schedule: OutputSchedule) {
        self.timeSpan = timeSpan
        self.schedule = schedule
        self.currentTime = timeSpan.start
    }
    
    @inlinable
    public mutating func nextTime(through: Double) -> Double? {
        switch schedule {
        case .everyAcceptedStep:
            return through
        case .times(let array):
            if currentTimeIndex >= array.count || array[currentTimeIndex] > through { return nil }
            let time = array[currentTimeIndex]
            currentTimeIndex &+= 1
            return time
        case .uniform(let step):
            if currentTime > through { return nil }
            let time = currentTime
            currentTime += step
            return time
        case .final:
            if through >= timeSpan.end { return timeSpan.end }
            return nil
        }
    }
}
