EigenTorch Documentation
========================

EigenTorch provides Eigen-style C++ APIs backed by libtorch for dense, sparse,
geometry, and unsupported-module workflows.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   quickstart
   module_coverage
   testing

API Reference
=============

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

Geometry types are provided through these headers:

- ``Eigen/Quaternion``
- ``Eigen/AngleAxis``
- ``Eigen/Translation``
- ``Eigen/Transform``
- ``Eigen/Rotation2D``
- ``Eigen/Scaling``
- ``Eigen/ParametrizedLine``
- ``Eigen/Hyperplane``
- ``Eigen/AlignedBox``

Sparse Operations
-----------------

Sparse APIs are provided through these headers:

- ``Eigen/SparseMatrix``
- ``Eigen/SparseVector``
- ``Eigen/SparseLU``
- ``Eigen/SparseQR``
- ``Eigen/SparseCholesky``

Jacobi Internal Iterators
-------------------------

Jacobi helpers are provided through ``Eigen/Jacobi``.
