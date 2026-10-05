/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, 2017, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
// meganan adaptation, 2026-10-05: minimal RGB/BGR uint8 bilinear resize
// with the same fixed-point rounding as OpenCV 5.0.0 INTER_LINEAR.
// Source: https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/resize.cpp
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace motion {
struct Letterbox {
    double scale;
    int width, height, left, top;
};

inline Letterbox resize_movenet(const std::uint8_t* source, int width, int height,
                                bool bgr, std::uint8_t* destination) {
    if (width <= 0 || height <= 0 || width > 16384 || height > 16384) {
        throw std::invalid_argument("Invalid frame dimensions");
    }
    const double scale = std::min(192.0 / width, 192.0 / height);
    const int resized_width = static_cast<int>(std::nearbyint(width * scale));
    const int resized_height = static_cast<int>(std::nearbyint(height * scale));
    if (resized_width <= 0 || resized_height <= 0) {
        throw std::invalid_argument("Frame aspect ratio is too extreme");
    }
    const Letterbox box{scale, resized_width, resized_height,
                        (192 - resized_width) / 2, (192 - resized_height) / 2};
    std::memset(destination, 0, 192 * 192 * 3);
    const double sx_scale = static_cast<double>(width) / resized_width;
    const double sy_scale = static_cast<double>(height) / resized_height;
    for (int y = 0; y < resized_height; ++y) {
        float fy = static_cast<float>((y + 0.5) * sy_scale - 0.5);
        int sy = static_cast<int>(std::floor(fy));
        fy -= sy;
        const int y0 = std::clamp(sy, 0, height - 1);
        const int y1 = std::clamp(sy + 1, 0, height - 1);
        const int b0 = static_cast<int>(std::lrint((1.f - fy) * 2048.f));
        const int b1 = static_cast<int>(std::lrint(fy * 2048.f));
        for (int x = 0; x < resized_width; ++x) {
            float fx = static_cast<float>((x + 0.5) * sx_scale - 0.5);
            int sx = static_cast<int>(std::floor(fx));
            fx -= sx;
            if (sx < 0) { sx = 0; fx = 0; }
            if (sx >= width - 1) { sx = width - 1; fx = 0; }
            const int x1 = std::min(sx + 1, width - 1);
            const int a0 = static_cast<int>(std::lrint((1.f - fx) * 2048.f));
            const int a1 = static_cast<int>(std::lrint(fx * 2048.f));
            auto* out = destination + ((box.top + y) * 192 + box.left + x) * 3;
            for (int c = 0; c < 3; ++c) {
                const int ch = bgr ? 2 - c : c;
                const int row0 = source[(y0 * width + sx) * 3 + ch] * a0 +
                                 source[(y0 * width + x1) * 3 + ch] * a1;
                const int row1 = source[(y1 * width + sx) * 3 + ch] * a0 +
                                 source[(y1 * width + x1) * 3 + ch] * a1;
                out[c] = static_cast<std::uint8_t>(
                    ((((b0 * (row0 >> 4)) >> 16) +
                      ((b1 * (row1 >> 4)) >> 16) + 2) >> 2));
            }
        }
    }
    return box;
}
}  // namespace motion
