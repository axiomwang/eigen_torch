.. EigenTorch documentation master file, created by
   sphinx-quickstart on Tue Mar 10 00:07:12 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

EigenTorch API Reference
========================

The following API exposes the comprehensive capabilities of the **Eigen** drop-in replacement library, seamlessly scaling matrix operations to massive GPU topologies via PyTorch.

.. toctree::
   :maxdepth: 2
   :caption: API Architecture:

Device Management
-----------------
.. doxygenclass:: Eigen::DeviceManager
   :members:

Core Linear Algebra
-------------------
.. doxygenclass:: Eigen::Matrix
   :members:

Solvers
-------
.. doxygenclass:: Eigen::PartialPivLU
   :members:
.. doxygenclass:: Eigen::LLT
   :members:
.. doxygenclass:: Eigen::HouseholderQR
   :members:
.. doxygenclass:: Eigen::JacobiSVD
   :members:
.. doxygenclass:: Eigen::EigenSolver
   :members:

Geometry
--------
.. doxygenclass:: Eigen::Quaternion
   :members:
.. doxygenclass:: Eigen::AngleAxis
   :members:
.. doxygenclass:: Eigen::Translation
   :members:
.. doxygenclass:: Eigen::Transform
   :members:
.. doxygenclass:: Eigen::Rotation2D
   :members:
.. doxygenclass:: Eigen::Scaling
   :members:
.. doxygenclass:: Eigen::ParametrizedLine
   :members:
.. doxygenclass:: Eigen::Hyperplane
   :members:
.. doxygenclass:: Eigen::AlignedBox
   :members:

Sparse Operations
-----------------
.. doxygenclass:: Eigen::SparseMatrix
   :members:
.. doxygenclass:: Eigen::SparseLU
   :members:
.. doxygenclass:: Eigen::SparseQR
   :members:
.. doxygenclass:: Eigen::SimplicialLLT
   :members:
.. doxygenclass:: Eigen::SimplicialLDLT
   :members:
   
Jacobi Internal Iterators
-------------------------
.. doxygenclass:: Eigen::JacobiRotation
   :members:
