# EigenTorch

EigenTorch is a header-only Eigen-compatible API layer backed by libtorch tensors.
It is designed for projects that want Eigen-style C++ APIs while executing calculations
through PyTorch (CPU, CUDA, or Apple MPS).

## What Is Implemented

- Core matrix/vector operations (`Eigen::Matrix`, dense arithmetic, blocks, maps, arrays)
- Dense decompositions and solvers (LU, QR, SVD, eigenvalue methods, LLT/LDLT)
- Geometry stack (`Quaternion`, `AngleAxis`, `Transform`, `Translation`, `Rotation2D`,
  `Scaling`, `ParametrizedLine`, `Hyperplane`, `AlignedBox`)
- Sparse stack (`SparseMatrix`, `SparseVector`, `SparseLU`, `SparseQR`,
  `SimplicialLLT`, `SimplicialLDLT`)
- Expanded unsupported modules with libtorch-backed calculations:
  - `AdolcForward`
  - `AlignedVector3`
  - `ArpackSupport`
  - `AutoDiff`
  - `BVH`
  - `EulerAngles`
  - `FFT`
  - `IterativeSolvers`
  - `KroneckerProduct`
  - `LevenbergMarquardt`
  - `MatrixFunctions`
  - `MPRealSupport`
  - `NNLS`
  - `NonLinearOptimization`
  - `NumericalDiff`
  - `OpenGLSupport`
  - `Polynomials`
  - `SparseExtra`
  - `SpecialFunctions`
  - `Splines`
  - `CXX11/Tensor`
  - `CXX11/TensorSymmetry`
  - `CXX11/ThreadPool`

## Build

From repository root:

```bash
cmake -S . -B build
cmake --build build -j8
```

## Install

```bash
cmake --install build
```

Then from a consumer CMake project:

```cmake
find_package(Torch REQUIRED)
find_package(Eigen REQUIRED)

target_link_libraries(my_app PRIVATE Eigen::Eigen)
```

## Basic Usage

```cpp
#include <Eigen/Dense>

int main() {
    Eigen::MatrixXf A = Eigen::MatrixXf::Random(3, 3);
    Eigen::VectorXf b = Eigen::VectorXf::Ones(3, 1);

    Eigen::VectorXf x = A.partialPivLu().solve(b);
    Eigen::MatrixXf C = A * A.inverse();
    (void)x;
    (void)C;
    return 0;
}
```

## Device Selection

Set the global default device before constructing matrices:

```cpp
#include <Eigen/Core>

int main() {
    if (torch::cuda::is_available()) {
        Eigen::DeviceManager::set_default_device(torch::kCUDA);
    } else if (Eigen::DeviceManager::is_mps_available()) {
        Eigen::DeviceManager::set_default_device(torch::kMPS);
    } else {
        Eigen::DeviceManager::set_default_device(torch::kCPU);
    }
    return 0;
}
```

## Test Suite

Run all tests:

```bash
cd build
ctest --output-on-failure
```

Current test targets:

- `core_test`
- `sparse_linalg_test`
- `api_compat_test`
- `unsupported_compat_test`
- `unsupported_full_modules_test`

## Documentation

Sphinx sources are in `docs/`.

Generate docs:

```bash
cd docs
doxygen Doxyfile
make html
```

Main page after build: `docs/_build/html/index.html`.
