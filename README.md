# EigenTorch (Drop-in Eigen API with PyTorch Backend)

This library serves as a computationally accelerated, drop-in replacement for standard [Eigen](http://eigen.tuxfamily.org/). It exposes the exact same syntax, headers, and namespaces (`Eigen::MatrixXf`, `#include <Eigen/Dense>`, etc.) but implements all math logic securely over PyTorch `at::Tensor` objects. This allows leveraging massive tensor optimization and heterogeneous hardware (like CUDA or Apple Silicon MPS) seamlessly within legacy Eigen-dependent C++ systems.

## 📦 Installation & CMake Discovery

We provide full export configurations for CMake so building and distributing is extremely easy.

### 1. Build & Install Locally
Compile the standalone library and install its headers into your standard system paths (or a prefix of your choosing):

```bash
mkdir build && cd build
cmake .. 
make install
```

### 2. Finding the Library in Your Project
In your consuming project's `CMakeLists.txt`, locate the library via `find_package`. The library is installed as CMake module `Eigen` and exports the target `Eigen::Eigen`:

```cmake
cmake_minimum_required(VERSION 3.14)
project(MyApp)

# Essential: Eigen depends on LibTorch
find_package(Torch REQUIRED)

# Find the installed header-only library
find_package(Eigen REQUIRED)

add_executable(my_app main.cpp)

# Link the dropped-in target
target_link_libraries(my_app PRIVATE Eigen::Eigen)
```

## 🚀 Usage & Device Selection

Include standard Eigen headers identically as you would traditionally.

```cpp
#include <iostream>
#include <Eigen/Dense>

int main() {
    // 1. Initializations
    Eigen::MatrixXf a = Eigen::MatrixXf::Random(3, 3);
    Eigen::MatrixXf b = Eigen::MatrixXf::Ones(3, 3);

    // Standard Math behaves perfectly
    Eigen::MatrixXf c = a * b;
    std::cout << c.mean() << std::endl;

    // Advanced features 
    Eigen::PartialPivLU<Eigen::MatrixXf> lu(a);
    Eigen::VectorXf x = lu.solve(Eigen::VectorXf::Ones(3, 1));
}
```

### 🖥 Choosing the Device (CPU vs GPU)

By default, the internal **DeviceManager** places all tensors on the **CPU** transparently. 

If you wish to offload computations to a GPU (CUDA) or an Apple Silicon MPS core natively, you simply need to configure the global pointer identically *before* instantiating any Eigen matrices.

```cpp
#include <Eigen/Core>

int main() {
    // Optional: Globally reroute Eigen mathematical allocations to CUDA natively
    if (torch::cuda::is_available()) {
        Eigen::DeviceManager::set_default_device(torch::kCUDA);
        std::cout << "Accelerating Eigen with CUDA." << std::endl;
    } 
    // Or MPS on Apple Silicon
    else if (Eigen::DeviceManager::is_mps_available()) {
        Eigen::DeviceManager::set_default_device(torch::kMPS);
        std::cout << "Accelerating Eigen with MPS." << std::endl;
    }
    else {
        // Defaults to torch::kCPU otherwise
        Eigen::DeviceManager::set_default_device(torch::kCPU);
    }

    // Now, all subsequent allocations and mathematics pipeline exactly to the configured hardware!
    Eigen::MatrixXf data = Eigen::MatrixXf::Random(1000, 1000); 
    Eigen::MatrixXf res = data * data.inverse(); 
}
```
