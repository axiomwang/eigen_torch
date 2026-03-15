Testing
=======

Run All Tests
-------------

From repository root:

.. code-block:: bash

   cmake -S . -B build
   cmake --build build -j8
   cd build
   ctest --output-on-failure

Current Test Targets
--------------------

- core_test
- sparse_linalg_test
- api_compat_test
- unsupported_compat_test
- unsupported_full_modules_test

What The Unsupported Tests Cover
--------------------------------

- unsupported_compat_test: broad compatibility and numerical checks across the original unsupported module set.
- unsupported_full_modules_test: deterministic numerical checks for the expanded unsupported module set, including optimization, iterative solvers, special functions, and CXX11 utilities.
