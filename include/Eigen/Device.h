#pragma once

#include <torch/torch.h>

namespace Eigen {

class DeviceManager {
public:
    // Global default device, defaults to CPU
    static inline torch::Device default_device = torch::kCPU;

    // Set the global default device
    static void set_default_device(const torch::Device& device) {
        default_device = device;
    }

    // Get the global default device
    static torch::Device get_default_device() {
        return default_device;
    }
    
    // Check if MPS (Metal Performance Shaders) is available on macOS
    static bool is_mps_available() {
        return torch::mps::is_available();
    }
};

} // namespace Eigen
