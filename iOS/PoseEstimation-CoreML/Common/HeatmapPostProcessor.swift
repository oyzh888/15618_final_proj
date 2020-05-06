//
//  HeatmapPostProcessor.swift
//  PoseEstimation-CoreML
//
//  Created by Doyoung Gwak on 27/06/2019.
//  Copyright © 2019 tucan9389. All rights reserved.
//

import Foundation
import CoreML
import simd

class HeatmapPostProcessor {
    
    //var maxvalue: Double = 0.5
    //var minvalue: Double = 0.5
    
    var onlyBust: Bool = false
//    let queue = DispatchQueue(label: "heatmap")
    var jointViewTime: Double = 0.0
    var heatmapTime: Double = 0.0
    var jointViewIterations: Int = 0
    var heatmapViewIterations: Int = 0
    
    func convertToPredictedPoints(from heatmaps: MLMultiArray, isFlipped: Bool = false) -> [PredictedPoint?] {
        guard heatmaps.shape.count >= 3 else {
            print("heatmap's shape is invalid. \(heatmaps.shape)")
            return []
        }
        let total_keypoint_number = heatmaps.shape[0].intValue
        var keypoint_number = total_keypoint_number
        if onlyBust { keypoint_number = min(total_keypoint_number, 8/*the index of R hip*/) }
        let heatmap_w = heatmaps.shape[1].intValue
        let heatmap_h = heatmaps.shape[2].intValue

        let start = DispatchTime.now()
        // Update the keypoints in parallel
        // UnsafeMutablePointer is similar to malloc in c
//        let keypoints_unsafe = UnsafeMutablePointer<PredictedPoint?>.allocate(capacity: keypoint_number)
        var n_kpoints = (0..<total_keypoint_number).map { _ -> PredictedPoint? in
                    return nil
        }
        
        // Optimize
        let _ = DispatchQueue.global(qos: .userInteractive)
        let heatmap_size = heatmap_h * heatmap_w
        let stride_size = 1
        DispatchQueue.concurrentPerform(iterations: keypoint_number/stride_size) { (k) in
//            let start_index = k*stride_size
//            let end_index = min(k*stride_size+stride_size, keypoint_number)
//            for kk in start_index..<end_index {
//                var max_confidence = 0.0
//                var max_i = 0
//                var max_j = 0
//                for i in 0..<heatmap_w {
//                    for j in 0..<heatmap_h {
//                        let index = kk*heatmap_size + i*(heatmap_h) + j
//                        let confidence = heatmaps[index].doubleValue
//                        guard confidence > 0  else { continue }
//                        if (max_confidence < confidence) {
//                            max_j = j
//                            max_i = i
//                            max_confidence = confidence
//                        }
//                    }
//                }
//
//                n_kpoints[kk] = PredictedPoint(maxPoint: CGPoint(x: CGFloat(max_j), y: CGFloat(max_i)), maxConfidence: max_confidence)
//            }

            //SIMD
            var max_idx = 0
            var max_v = simd_make_double4(0.0)
            var max_confidence = 0.0
            let base_index = k*heatmap_size

            // A12 chip 128 KB L1 cache
            let vecArray = stride(from: base_index, to: base_index+heatmap_size, by: 4)
                .map { (i:Int) -> simd_double4 in
                    simd_double4(heatmaps[i].doubleValue,
                        heatmaps[i+1].doubleValue,
                        heatmaps[i+2].doubleValue,
                        heatmaps[i+3].doubleValue)
            }

            for idx in 0..<vecArray.count {
                let v = vecArray[idx]
                let confidence = simd_reduce_max(v)
                if (max_confidence < confidence) {
                    max_confidence = confidence
                    max_v = v
                    max_idx = idx
                }
            }

            var max_j = max_idx << 2 % heatmap_h
            let max_i = max_idx << 2 / heatmap_h

            if (Double(max_v[1]) == max_confidence) {
                max_j += 1
            } else if (Double(max_v[2]) == max_confidence) {
                max_j += 2
            } else if (Double(max_v[3]) == max_confidence) {
                max_j += 3
            }

            n_kpoints[k] = PredictedPoint(maxPoint: CGPoint(x: CGFloat(max_j), y: CGFloat(max_i)), maxConfidence: max_confidence)
        }

          // Original
//        for k in 0..<keypoint_number {
//            for i in 0..<heatmap_w {
//                for j in 0..<heatmap_h {
//                    let index = k*(heatmap_w*heatmap_h) + i*(heatmap_h) + j
//                    let confidence = heatmaps[index].doubleValue
//                    //if maxvalue < confidence { maxvalue = confidence }
//                    //if minvalue > confidence { minvalue = confidence }
//                    guard confidence > 0  else { continue }
//                    if n_kpoints[k] == nil ||
//                        (n_kpoints[k] != nil && n_kpoints[k]!.maxConfidence < confidence) {
//                        n_kpoints[k] = PredictedPoint(maxPoint: CGPoint(x: CGFloat(j), y: CGFloat(i)), maxConfidence: confidence)
//                    }
//                }
//            }
//        }
        
        let end = DispatchTime.now()
        let nanoTime = end.uptimeNanoseconds - start.uptimeNanoseconds
        let timeInterval = Double(nanoTime) / 1_000_000
        jointViewTime += timeInterval
        jointViewIterations += 1
        if (jointViewIterations % 300 == 0) {
            jointViewTime = 0.0
            jointViewIterations = 0
            print("Time to heatmap postprocessing \(timeInterval) ms on average (100 frames)")
        }
        
        // transpose to (1.0, 1.0)
        n_kpoints = n_kpoints.map { kpoint -> PredictedPoint? in
            if let kp = kpoint {
                var x: CGFloat = (kp.maxPoint.x+0.5)/CGFloat(heatmap_w)
                let y: CGFloat = (kp.maxPoint.y+0.5)/CGFloat(heatmap_h)
                if isFlipped { x = 1 - x }
                return PredictedPoint(maxPoint: CGPoint(x: x, y: y),
                                      maxConfidence: kp.maxConfidence)
            } else {
                return nil
            }
        }
        
        return n_kpoints
    }
    
    func convertTo2DArray(from heatmaps: MLMultiArray) -> Array<Array<Double>> {
         guard heatmaps.shape.count >= 3 else {
             print("heatmap's shape is invalid. \(heatmaps.shape)")
             return []
         }
        
         let start = DispatchTime.now()
         let keypoint_number = heatmaps.shape[0].intValue
         let heatmap_w = heatmaps.shape[1].intValue
         let heatmap_h = heatmaps.shape[2].intValue
         var convertedHeatmap: Array<Array<Double>> = Array(repeating: Array(repeating: 0.0, count: heatmap_h), count: heatmap_w)
         
         // Original
//        for k in 0..<keypoint_number {
//            for y in 0..<heatmap_w {
//                for x in 0..<heatmap_h {
//                    let index = k*(heatmap_w*heatmap_h) + y*(heatmap_h) + x
//                    let confidence = heatmaps[index].doubleValue
//                    guard confidence > 0 else { continue }
//                    convertedHeatmap[x][y] += confidence
//                }
//            }
//        }
        
         // Optimized
         let _ = DispatchQueue.global(qos: .userInteractive)
         let heatmap_size = heatmap_h*heatmap_w
         let convertedHeatmap_1d = UnsafeMutablePointer<Double>.allocate(capacity: heatmap_size)
         DispatchQueue.concurrentPerform(iterations: heatmap_size) { (i) in
            // SIMD
//            var confidence = simd_make_double4(0.0)
//            for k in stride(from: 0, through: keypoint_number, by: 4) {
//                let index1 = k*heatmap_size + i
//                let index2 = (k+1)*heatmap_size + i
//                let index3 = (k+2)*heatmap_size + i
//                let index4 = (k+3)*heatmap_size + i
//                if keypoint_number - k >= 4 {
//                    confidence = confidence + simd_double4(heatmaps[index1].doubleValue, heatmaps[index2].doubleValue,
//                    heatmaps[index3].doubleValue, heatmaps[index4].doubleValue)
//                } else {
//                    confidence = confidence + simd_double4(heatmaps[k].doubleValue, heatmaps[k+1].doubleValue, 0.0, 0.0)
//                }
//            }
//            convertedHeatmap_1d[i] = reduce_add(confidence)
            var total_confidence = 0.0
            for k in 0..<keypoint_number {
                 let index = k*heatmap_size + i
                 let confidence = heatmaps[index].doubleValue
                 guard confidence > 0 else { continue }
                 total_confidence += confidence
            }
            convertedHeatmap_1d[i] = total_confidence
         }

        for x in 0..<heatmap_w {
            for y in 0..<heatmap_h {
                let index = y*(heatmap_h) + x
                convertedHeatmap[x][y] = convertedHeatmap_1d[index]
            }
        }
         
         convertedHeatmap = convertedHeatmap.map { row in
             return row.map { element in
                 if element > 1.0 {
                     return 1.0
                 } else if element < 0 {
                     return 0.0
                 } else {
                     return element
                 }
             }
         }
        let end = DispatchTime.now()
        let nanoTime = end.uptimeNanoseconds - start.uptimeNanoseconds
        let timeInterval = Double(nanoTime) / 1_000_000
        heatmapTime += timeInterval
        heatmapViewIterations += 1
        if (heatmapViewIterations % 300 == 0) {
            heatmapTime = 0.0
            heatmapViewIterations = 0
            print("Time to heatmap postprocessing (heatmap mode) \(timeInterval) ms on average (100 frames)")
        }
         
         return convertedHeatmap
     }
}
