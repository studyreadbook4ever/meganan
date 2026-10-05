#pragma once

#include <cstdint>
#include <vector>

namespace motion {

// Camera pixels stay inside the native engine. Python receives only this
// object's dimensions and passes its shared ownership to native inference.
struct Frame {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgb;
};

}  // namespace motion
