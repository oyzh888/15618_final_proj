//
//  MovingAverageFilter.swift
//  PoseEstimation-CoreML
//
//  Created by Doyoung Gwak on 26/06/2019.
//  Copyright © 2019 tucan9389. All rights reserved.
//

import UIKit

extension CGPoint {
    static func +(lhs: CGPoint, rhs: CGPoint) -> CGPoint {
        return CGPoint(x: lhs.x + rhs.x, y: lhs.y + rhs.y)
    }
    
    static func /(lhs: CGPoint, rhs: CGFloat) -> CGPoint {
        guard rhs != 0.0 else { return lhs }
        return CGPoint(x: lhs.x / rhs, y: lhs.y / rhs)
    }
}

class MovingAverageFilter {
    var elements: [PredictedPoint?] = []
    private var limit: Int

    init(limit: Int) {
        guard limit > 0 else { fatalError("limit should be uppered than 0 in MovingAverageFilter init(limit:)") }
        self.elements = []
        self.limit = limit
    }

    func add(element: PredictedPoint?) {
        elements.append(element)
        while self.elements.count > self.limit {
            self.elements.removeFirst()
        }
    }

    func averagedValue() -> PredictedPoint? {
        let nonoptionalPoints: [CGPoint] = elements.compactMap{ $0?.maxPoint }
        let nonoptionalConfidences: [Double] = elements.compactMap{ $0?.maxConfidence }
        guard !nonoptionalPoints.isEmpty && !nonoptionalConfidences.isEmpty else { return nil }
        let sumPoint = nonoptionalPoints.reduce( CGPoint.zero ) { $0 + $1 }
        let sumConfidence = nonoptionalConfidences.reduce( 0.0 ) { $0 + $1 }
        return PredictedPoint(maxPoint: sumPoint / CGFloat(nonoptionalPoints.count), maxConfidence: sumConfidence)
    }
}

//class MovingAverageFilter {
//    var elements: [PredictedPoint?] = []
//    private var limit: Int
//
//    init(limit: Int) {
//        guard limit > 0 else { fatalError("limit should be uppered than 0 in MovingAverageFilter init(limit:)") }
//        self.elements = []
//        self.limit = limit
//    }
//
//    func add(element: PredictedPoint?) {
//        elements.append(element)
//        while self.elements.count > self.limit {
//            self.elements.removeFirst()
//        }
//    }
//
//    func calculateMedian(array: [CGFloat]) -> CGFloat {
//        let sorted = array.sorted()
//        if sorted.count % 2 == 0 {
//            return CGFloat((sorted[(sorted.count / 2)] + sorted[(sorted.count / 2) - 1])) / 2
//        } else {
//            return CGFloat(sorted[(sorted.count - 1) / 2])
//        }
//    }
//
//    func averagedValue() -> PredictedPoint? {
//        let nonoptionalPoints: [CGPoint] = elements.compactMap{ $0?.maxPoint }
//        let nonoptionalConfidences: [Double] = elements.compactMap{ $0?.maxConfidence }
//        guard !nonoptionalPoints.isEmpty && !nonoptionalConfidences.isEmpty else { return nil }
//
//        var xArrs: [CGFloat] = []
//        var yArrs: [CGFloat] = []
//        for p in nonoptionalPoints{
//            xArrs.append(p.x)
//            yArrs.append(p.y)
//        }
//        let sumConfidence = nonoptionalConfidences.reduce( 0.0 ) { $0 + $1 }
////        let sumPoint = nonoptionalPoints.reduce( CGPoint.zero ) { $0 + $1 }
////        return PredictedPoint(maxPoint: sumPoint / CGFloat(nonoptionalPoints.count), maxConfidence: sumConfidence)
//        let xMedian = calculateMedian(array:  xArrs)
//        let yMedian = calculateMedian(array: yArrs)
//        return PredictedPoint(maxPoint: CGPoint(x: xMedian, y: yMedian), maxConfidence: sumConfidence)
////        return CGPoint(x: lhs.x / rhs, y: lhs.y / rhs)
//
//    }
//}
