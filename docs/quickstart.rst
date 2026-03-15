Quickstart
==========

Requirements
------------

- CMake 3.25+
- C++17 compiler
- LibTorch (CPU or CUDA build)
- Python torch package if you rely on automatic Torch discovery in CMake

Build
-----

From repository root:

.. code-block:: bash

   cmake -S . -B build
   cmake --build build -j8

Select Compute Device
---------------------

EigenTorch uses Eigen-style APIs on top of torch tensors. You can select the default device globally before constructing matrices.

.. code-block:: cpp

   #include <Eigen/Core>

   int main() {
       if (torch::cuda::is_available()) {
           Eigen::DeviceManager::set_default_device(torch::kCUDA);
       } else if (Eigen::DeviceManager::is_mps_available()) {
           Eigen::DeviceManager::set_default_device(torch::kMPS);
       } else {
           Eigen::DeviceManager::set_default_device(torch::kCPU);
       }
   }

Example
-------

.. code-block:: cpp

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
