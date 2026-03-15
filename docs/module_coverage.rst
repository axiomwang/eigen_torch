Module Coverage
===============

Core and Dense
--------------

The project provides Eigen-style headers and matrix/vector APIs backed by libtorch tensors.

- Core matrix/vector operations
- Decompositions and solvers (LU, QR, SVD, eigenvalue routines)
- Geometry stack (Quaternion, AngleAxis, Transform, Translation, Rotation2D, Scaling, ParametrizedLine, Hyperplane, AlignedBox)

Sparse
------

Sparse APIs are implemented with Eigen-compatible lifecycle methods and torch-backed math, including dense fallbacks for unsupported sparse kernels.

- SparseMatrix and SparseVector
- SparseLU
- SparseQR
- SimplicialLLT
- SimplicialLDLT

Unsupported Modules
-------------------

The following unsupported modules are currently implemented and tested:

- AdolcForward
- AlignedVector3
- ArpackSupport
- AutoDiff
- BVH
- EulerAngles
- FFT
- IterativeSolvers
- KroneckerProduct
- LevenbergMarquardt
- MatrixFunctions
- MPRealSupport
- NNLS
- NonLinearOptimization
- NumericalDiff
- OpenGLSupport
- Polynomials
- SparseExtra
- SpecialFunctions
- Splines
- CXX11/Tensor
- CXX11/TensorSymmetry
- CXX11/ThreadPool
