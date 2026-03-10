#pragma once

#include <torch/torch.h>
#include "Device.h"
#include <iostream>

namespace Eigen {

// Forward declarations equivalent to Eigen's Dynamic
constexpr int Dynamic = -1;

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
class Matrix {
protected:
    at::Tensor m_tensor;

public:
    // Expose scalar type strictly under different alias to avoid template shadowing
    using ScalarType = Scalar;

    // Helper map to get Torch scalar types
    static constexpr torch::ScalarType get_torch_dtype() {
        if constexpr (std::is_same_v<Scalar, float>) return torch::kFloat32;
        if constexpr (std::is_same_v<Scalar, double>) return torch::kFloat64;
        if constexpr (std::is_same_v<Scalar, int32_t>) return torch::kInt32;
        if constexpr (std::is_same_v<Scalar, int64_t>) return torch::kInt64;
        return torch::kFloat32; // Default
    }

    // Default constructor (uninitialized tensor)
    Matrix() : m_tensor() {}

    // Explicit constructor from an existing at::Tensor
    explicit Matrix(const at::Tensor& tensor) : m_tensor(tensor) {}

    // Constructor with dimensions
    Matrix(int rows, int cols) {
        auto options = torch::TensorOptions()
            .dtype(get_torch_dtype())
            .device(DeviceManager::get_default_device());
        m_tensor = torch::empty({rows, cols}, options);
    }
    
    // Copy constructor/Converting constructor from another Matrix
    template <typename OtherScalar, int OtherRows, int OtherCols>
    Matrix(const Matrix<OtherScalar, OtherRows, OtherCols>& other) {
        // Technically we should check dimensions if both are static
        m_tensor = other.tensor();
    }
    
    // Provide access to the underlying tensor if needed by adv users
    at::Tensor& tensor() { return m_tensor; }
    const at::Tensor& tensor() const { return m_tensor; }

    // Shape info
    int rows() const { return m_tensor.defined() ? m_tensor.size(0) : 0; }
    int cols() const { return m_tensor.defined() ? m_tensor.size(1) : 0; }
    int size() const { return m_tensor.defined() ? m_tensor.numel() : 0; }

    // Element access (Note: this is notoriously slow in libtorch if used in tight loops, parity with Eigen's operator())
    Scalar operator()(int row, int col) const {
        return m_tensor[row][col].item<Scalar>();
    }

    // Static initializers
    static Matrix Random(int rows, int cols) {
        auto options = torch::TensorOptions()
            .dtype(get_torch_dtype())
            .device(DeviceManager::get_default_device());
        return Matrix(torch::rand({rows, cols}, options));
    }

    static Matrix Zero(int rows, int cols) {
        auto options = torch::TensorOptions()
            .dtype(get_torch_dtype())
            .device(DeviceManager::get_default_device());
        return Matrix(torch::zeros({rows, cols}, options));
    }

    static Matrix Ones(int rows, int cols) {
        auto options = torch::TensorOptions()
            .dtype(get_torch_dtype())
            .device(DeviceManager::get_default_device());
        return Matrix(torch::ones({rows, cols}, options));
    }

    // Initialize with Identity
    static Matrix Identity(int rows, int cols) {
        auto options = torch::TensorOptions()
            .dtype(get_torch_dtype())
            .device(DeviceManager::get_default_device());
        return Matrix(torch::eye(rows, cols, options));
    }
    
    // Matrix Multiplication (mimicking Eigen's operator*)
    template <typename OtherScalar, int OtherRows, int OtherCols>
    Matrix<Scalar, Dynamic, Dynamic> operator*(const Matrix<OtherScalar, OtherRows, OtherCols>& other) const {
        // Torch does mm (matrix multiplication)
        at::Tensor result = torch::matmul(m_tensor, other.tensor());
        return Matrix<Scalar, Dynamic, Dynamic>(result);
    }

    // Element-wise arithmetic
    template <typename OtherScalar, int OtherRows, int OtherCols>
    Matrix<Scalar, Dynamic, Dynamic> operator+(const Matrix<OtherScalar, OtherRows, OtherCols>& other) const {
        return Matrix<Scalar, Dynamic, Dynamic>(m_tensor + other.tensor());
    }

    template <typename OtherScalar, int OtherRows, int OtherCols>
    Matrix<Scalar, Dynamic, Dynamic> operator-(const Matrix<OtherScalar, OtherRows, OtherCols>& other) const {
        return Matrix<Scalar, Dynamic, Dynamic>(m_tensor - other.tensor());
    }

    // Transpose
    Matrix<Scalar, Dynamic, Dynamic> transpose() const {
        return Matrix<Scalar, Dynamic, Dynamic>(m_tensor.transpose(0, 1));
    }

    // Adjoint (conjugate transpose)
    Matrix<Scalar, Dynamic, Dynamic> adjoint() const {
        // Since we primarily deal with real numbers for now, adjoint == transpose. If complex, would need conjunct.
        // For PyTorch: at::conj(m_tensor).transpose(0, 1) if complex, but adjoint() in Torch does that natively.
        return Matrix<Scalar, Dynamic, Dynamic>(m_tensor.adjoint());
    }

    // Dot product
    template <typename OtherScalar, int OtherRows, int OtherCols>
    Scalar dot(const Matrix<OtherScalar, OtherRows, OtherCols>& other) const {
        // In Eigen, dot is generally for vectors. We will flatten and dot to match behavior or just use torch::dot 
        // Note: torch::dot only supports 1D tensors.
        return torch::dot(m_tensor.reshape({-1}), other.tensor().reshape({-1})).template item<Scalar>();
    }

    // Block operations (creates views)
    Matrix<Scalar, Dynamic, Dynamic> block(int startRow, int startCol, int blockRows, int blockCols) {
        return Matrix<Scalar, Dynamic, Dynamic>(
            m_tensor.slice(0, startRow, startRow + blockRows)
                    .slice(1, startCol, startCol + blockCols)
        );
    }
    
    // Const block operations
    const Matrix<Scalar, Dynamic, Dynamic> block(int startRow, int startCol, int blockRows, int blockCols) const {
        return Matrix<Scalar, Dynamic, Dynamic>(
            m_tensor.slice(0, startRow, startRow + blockRows)
                    .slice(1, startCol, startCol + blockCols)
        );
    }

    // Row extraction
    Matrix<Scalar, 1, Dynamic> row(int i) {
        return Matrix<Scalar, 1, Dynamic>(m_tensor.slice(0, i, i + 1));
    }

    const Matrix<Scalar, 1, Dynamic> row(int i) const {
        return Matrix<Scalar, 1, Dynamic>(m_tensor.slice(0, i, i + 1));
    }

    // Column extraction
    Matrix<Scalar, Dynamic, 1> col(int i) {
        return Matrix<Scalar, Dynamic, 1>(m_tensor.slice(1, i, i + 1));
    }

    const Matrix<Scalar, Dynamic, 1> col(int i) const {
        return Matrix<Scalar, Dynamic, 1>(m_tensor.slice(1, i, i + 1));
    }

    // Element-wise operations
    Matrix<Scalar, Dynamic, Dynamic> cwiseProduct(const Matrix& other) const {
        return Matrix<Scalar, Dynamic, Dynamic>(m_tensor * other.tensor());
    }

    Matrix<Scalar, Dynamic, Dynamic> cwiseQuotient(const Matrix& other) const {
        return Matrix<Scalar, Dynamic, Dynamic>(m_tensor / other.tensor());
    }
    
    Matrix<Scalar, Dynamic, Dynamic> cwiseAbs() const {
        return Matrix<Scalar, Dynamic, Dynamic>(torch::abs(m_tensor));
    }

    // Reductions
    Scalar sum() const {
        return m_tensor.sum().item<Scalar>();
    }

    Scalar mean() const {
        return m_tensor.mean().item<Scalar>();
    }

    Scalar maxCoeff() const {
        return m_tensor.max().item<Scalar>();
    }

    Scalar minCoeff() const {
        return m_tensor.min().item<Scalar>();
    }

    // Linear Algebra
    Matrix<Scalar, Dynamic, Dynamic> inverse() const {
        // Note: torch::inverse expects float or double
        return Matrix<Scalar, Dynamic, Dynamic>(torch::inverse(m_tensor));
    }

    // Stream operator for printing
    friend std::ostream& operator<<(std::ostream& os, const Matrix& m) {
        os << m.tensor();
        return os;
    }
};

// Common dynamic typedefs
using MatrixXf = Matrix<float, Dynamic, Dynamic>;
using MatrixXd = Matrix<double, Dynamic, Dynamic>;
using VectorXf = Matrix<float, Dynamic, 1>;
using VectorXd = Matrix<double, Dynamic, 1>;

// Fixed size aliases (using dynamic underneath for LibTorch compatibility but with constrained constructors later, matching standard syntax here)
using Matrix2f = Matrix<float, 2, 2>;
using Matrix3f = Matrix<float, 3, 3>;
using Matrix4f = Matrix<float, 4, 4>;

using Matrix2d = Matrix<double, 2, 2>;
using Matrix3d = Matrix<double, 3, 3>;
using Matrix4d = Matrix<double, 4, 4>;

using Vector2f = Matrix<float, 2, 1>;
using Vector3f = Matrix<float, 3, 1>;
using Vector4f = Matrix<float, 4, 1>;

using Vector2d = Matrix<double, 2, 1>;
using Vector3d = Matrix<double, 3, 1>;
using Vector4d = Matrix<double, 4, 1>;

} // namespace Eigen
