#pragma once

#include <torch/torch.h>
#include "Device.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace Eigen {

// Forward declarations equivalent to Eigen's Dynamic
constexpr int Dynamic = -1;

// Forward declarations for decomposition chaining (defined in Dense)
template <typename MatrixType> class PartialPivLU;
template <typename MatrixType> class FullPivLU;
template <typename MatrixType> class HouseholderQR;
template <typename MatrixType> class ColPivHouseholderQR;
template <typename MatrixType> class FullPivHouseholderQR;
template <typename MatrixType> class LLT;
template <typename MatrixType> class LDLT;
template <typename MatrixType> class JacobiSVD;
template <typename MatrixType> class BDCSVD;
template <typename MatrixType> class CompleteOrthogonalDecomposition;
template <typename MatrixType> class EigenSolver;
template <typename MatrixType> class SelfAdjointEigenSolver;
template <typename MatrixType> class ComplexEigenSolver;
template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime> class Array;

template <typename Scalar_, int RowsAtCompileTime_, int ColsAtCompileTime_>
class Matrix {
protected:
    at::Tensor m_tensor;

    static at::TensorOptions default_options() {
        return torch::TensorOptions()
            .dtype(get_torch_dtype())
            .device(DeviceManager::get_default_device());
    }

    static void validate_resize(int rows, int cols) {
        if constexpr (RowsAtCompileTime_ != Dynamic) {
            if (rows != RowsAtCompileTime_) {
                throw std::invalid_argument("Row count does not match fixed compile-time dimension.");
            }
        }
        if constexpr (ColsAtCompileTime_ != Dynamic) {
            if (cols != ColsAtCompileTime_) {
                throw std::invalid_argument("Column count does not match fixed compile-time dimension.");
            }
        }
    }

public:
    using Scalar = Scalar_;

    // Expose scalar type strictly under different alias to avoid template shadowing
    using ScalarType = Scalar_;

    static constexpr int RowsAtCompileTime = RowsAtCompileTime_;
    static constexpr int ColsAtCompileTime = ColsAtCompileTime_;
    static constexpr int SizeAtCompileTime =
        (RowsAtCompileTime_ == Dynamic || ColsAtCompileTime_ == Dynamic)
            ? Dynamic
            : RowsAtCompileTime_ * ColsAtCompileTime_;

    class CoeffProxy {
    private:
        at::Tensor m_element;

    public:
        explicit CoeffProxy(const at::Tensor& element) : m_element(element) {}

        CoeffProxy& operator=(const Scalar_& value) {
            m_element.fill_(value);
            return *this;
        }

        template <typename T>
        CoeffProxy& operator=(const T& value) {
            m_element.fill_(static_cast<Scalar_>(value));
            return *this;
        }

        operator Scalar_() const {
            return m_element.template item<Scalar_>();
        }
    };

    // Helper map to get Torch scalar types
    static constexpr torch::ScalarType get_torch_dtype() {
        if constexpr (std::is_same_v<Scalar_, float>) return torch::kFloat32;
        if constexpr (std::is_same_v<Scalar_, double>) return torch::kFloat64;
        if constexpr (std::is_same_v<Scalar_, int32_t>) return torch::kInt32;
        if constexpr (std::is_same_v<Scalar_, int64_t>) return torch::kInt64;
        if constexpr (std::is_same_v<Scalar_, std::complex<float>>) return torch::kComplexFloat;
        if constexpr (std::is_same_v<Scalar_, std::complex<double>>) return torch::kComplexDouble;
        return torch::kFloat32; // Default
    }

    // Default constructor (uninitialized tensor)
    Matrix() : m_tensor() {}

    // Explicit constructor from an existing at::Tensor
    explicit Matrix(const at::Tensor& tensor) : m_tensor(tensor) {
        if (m_tensor.defined() && m_tensor.dim() == 1) {
            m_tensor = m_tensor.unsqueeze(1);
        }
    }

    // Constructor with dimensions
    Matrix(int rows, int cols) {
        resize(rows, cols);
    }

    // Vector-like constructor
    explicit Matrix(int size) {
        if constexpr (RowsAtCompileTime_ == 1 && ColsAtCompileTime_ != 1) {
            resize(1, size);
        } else {
            resize(size, 1);
        }
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
    int rows() const { return m_tensor.defined() ? static_cast<int>(m_tensor.size(0)) : 0; }
    int cols() const { return m_tensor.defined() ? static_cast<int>(m_tensor.size(1)) : 0; }
    int size() const { return m_tensor.defined() ? m_tensor.numel() : 0; }
    bool isVector() const { return rows() == 1 || cols() == 1; }

    // Raw data access (CPU contiguous tensors only)
    Scalar_* data() {
        if (!m_tensor.defined()) return nullptr;
        if (!m_tensor.is_cpu() || !m_tensor.is_contiguous()) {
            throw std::runtime_error("data() requires a contiguous CPU tensor.");
        }
        return m_tensor.data_ptr<Scalar_>();
    }

    const Scalar_* data() const {
        if (!m_tensor.defined()) return nullptr;
        if (!m_tensor.is_cpu() || !m_tensor.is_contiguous()) {
            throw std::runtime_error("data() requires a contiguous CPU tensor.");
        }
        return m_tensor.data_ptr<Scalar_>();
    }

    // Element access (Note: this is notoriously slow in libtorch if used in tight loops, parity with Eigen's operator())
    Scalar_ operator()(int row, int col) const {
        return m_tensor[row][col].template item<Scalar_>();
    }

    CoeffProxy operator()(int row, int col) {
        return CoeffProxy(m_tensor.index({row, col}));
    }

    // Vector element access
    Scalar_ operator()(int index) const {
        return m_tensor.reshape({-1})[index].template item<Scalar_>();
    }

    CoeffProxy operator()(int index) {
        return CoeffProxy(m_tensor.reshape({-1}).index({index}));
    }

    Scalar_ coeff(int row, int col) const {
        return operator()(row, col);
    }

    CoeffProxy coeffRef(int row, int col) {
        return operator()(row, col);
    }

    Scalar_ coeff(int index) const {
        return operator()(index);
    }

    CoeffProxy coeffRef(int index) {
        return operator()(index);
    }

    Scalar_ x() const { return operator()(0); }
    Scalar_ y() const { return operator()(1); }
    Scalar_ z() const { return operator()(2); }
    Scalar_ w() const { return operator()(3); }

    CoeffProxy x() { return operator()(0); }
    CoeffProxy y() { return operator()(1); }
    CoeffProxy z() { return operator()(2); }
    CoeffProxy w() { return operator()(3); }

    // Resizing
    void resize(int rows, int cols) {
        validate_resize(rows, cols);
        m_tensor = torch::empty({rows, cols}, default_options());
    }

    void resize(int size) {
        if constexpr (RowsAtCompileTime_ == 1 && ColsAtCompileTime_ != 1) {
            resize(1, size);
        } else {
            resize(size, 1);
        }
    }

    template <typename OtherScalar, int OtherRows, int OtherCols>
    void resizeLike(const Matrix<OtherScalar, OtherRows, OtherCols>& other) {
        resize(other.rows(), other.cols());
    }

    void conservativeResize(int rows, int cols) {
        validate_resize(rows, cols);
        if (!m_tensor.defined()) {
            resize(rows, cols);
            return;
        }

        at::Tensor resized = torch::zeros({rows, cols}, m_tensor.options());
        int copy_rows = std::min(rows, this->rows());
        int copy_cols = std::min(cols, this->cols());
        resized.slice(0, 0, copy_rows).slice(1, 0, copy_cols)
              .copy_(m_tensor.slice(0, 0, copy_rows).slice(1, 0, copy_cols));
        m_tensor = resized;
    }

    void conservativeResize(int size) {
        if constexpr (RowsAtCompileTime_ == 1 && ColsAtCompileTime_ != 1) {
            conservativeResize(1, size);
        } else {
            conservativeResize(size, 1);
        }
    }

    // Static initializers
    static Matrix Random(int rows, int cols) {
        if constexpr (std::is_integral_v<Scalar_>) {
            return Matrix(torch::randint(0, 100, {rows, cols}, default_options()));
        }
        return Matrix(torch::rand({rows, cols}, default_options()));
    }

    static Matrix Zero(int rows, int cols) {
        return Matrix(torch::zeros({rows, cols}, default_options()));
    }

    static Matrix Ones(int rows, int cols) {
        return Matrix(torch::ones({rows, cols}, default_options()));
    }

    // Initialize with Identity
    static Matrix Identity(int rows, int cols) {
        return Matrix(torch::eye(rows, cols, default_options()));
    }

    static Matrix Constant(int rows, int cols, Scalar_ value) {
        return Matrix(torch::full({rows, cols}, value, default_options()));
    }

    static Matrix Unit(int size, int index) {
        Matrix result = Zero(size, 1);
        result(index, 0) = static_cast<Scalar_>(1);
        return result;
    }

    static Matrix Unit(int index) {
        static_assert(RowsAtCompileTime_ != Dynamic || ColsAtCompileTime_ != Dynamic,
                      "Unit(index) requires fixed-size dimensions or use Unit(size, index).");

        constexpr int fixed_size =
            RowsAtCompileTime_ == 1 ? ColsAtCompileTime_ : RowsAtCompileTime_;
        Matrix result = Zero(
            RowsAtCompileTime_ == Dynamic ? fixed_size : RowsAtCompileTime_,
            ColsAtCompileTime_ == Dynamic ? fixed_size : ColsAtCompileTime_);

        if constexpr (RowsAtCompileTime_ == 1) {
            result(0, index) = static_cast<Scalar_>(1);
        } else {
            result(index, 0) = static_cast<Scalar_>(1);
        }
        return result;
    }

    static Matrix UnitX() { return Unit(0); }
    static Matrix UnitY() { return Unit(1); }
    static Matrix UnitZ() { return Unit(2); }
    static Matrix UnitW() { return Unit(3); }

    // In-place set helpers
    Matrix& setZero() {
        if (!m_tensor.defined()) {
            resize(RowsAtCompileTime_ == Dynamic ? 0 : RowsAtCompileTime_,
                   ColsAtCompileTime_ == Dynamic ? 0 : ColsAtCompileTime_);
        }
        m_tensor.zero_();
        return *this;
    }

    Matrix& setZero(int rows, int cols) {
        resize(rows, cols);
        m_tensor.zero_();
        return *this;
    }

    Matrix& setOnes() {
        m_tensor.fill_(1);
        return *this;
    }

    Matrix& setOnes(int rows, int cols) {
        resize(rows, cols);
        m_tensor.fill_(1);
        return *this;
    }

    Matrix& setConstant(Scalar_ value) {
        m_tensor.fill_(value);
        return *this;
    }

    Matrix& setConstant(int rows, int cols, Scalar_ value) {
        resize(rows, cols);
        m_tensor.fill_(value);
        return *this;
    }

    Matrix& setRandom() {
        if constexpr (std::is_integral_v<Scalar_>) {
            m_tensor.random_(0, 100);
        } else {
            m_tensor.uniform_();
        }
        return *this;
    }

    Matrix& setRandom(int rows, int cols) {
        resize(rows, cols);
        return setRandom();
    }

    Matrix& setIdentity() {
        if (!m_tensor.defined()) {
            throw std::runtime_error("setIdentity() requires an already sized matrix.");
        }
        m_tensor = torch::eye(rows(), cols(), m_tensor.options());
        return *this;
    }

    Matrix& setIdentity(int rows, int cols) {
        resize(rows, cols);
        m_tensor = torch::eye(rows, cols, m_tensor.options());
        return *this;
    }
    
    // Matrix Multiplication (mimicking Eigen's operator*)
    template <typename OtherScalar, int OtherRows, int OtherCols>
    Matrix<Scalar_, Dynamic, Dynamic> operator*(const Matrix<OtherScalar, OtherRows, OtherCols>& other) const {
        // Torch does mm (matrix multiplication)
        at::Tensor result = torch::matmul(m_tensor, other.tensor());
        return Matrix<Scalar_, Dynamic, Dynamic>(result);
    }

    // Element-wise arithmetic
    template <typename OtherScalar, int OtherRows, int OtherCols>
    Matrix<Scalar_, Dynamic, Dynamic> operator+(const Matrix<OtherScalar, OtherRows, OtherCols>& other) const {
        return Matrix<Scalar_, Dynamic, Dynamic>(m_tensor + other.tensor());
    }

    template <typename OtherScalar, int OtherRows, int OtherCols>
    Matrix<Scalar_, Dynamic, Dynamic> operator-(const Matrix<OtherScalar, OtherRows, OtherCols>& other) const {
        return Matrix<Scalar_, Dynamic, Dynamic>(m_tensor - other.tensor());
    }

    Matrix<Scalar_, Dynamic, Dynamic> operator*(Scalar_ scalar) const {
        return Matrix<Scalar_, Dynamic, Dynamic>(m_tensor * scalar);
    }

    Matrix<Scalar_, Dynamic, Dynamic> operator/(Scalar_ scalar) const {
        return Matrix<Scalar_, Dynamic, Dynamic>(m_tensor / scalar);
    }

    Matrix<Scalar_, Dynamic, Dynamic> operator+(Scalar_ scalar) const {
        return Matrix<Scalar_, Dynamic, Dynamic>(m_tensor + scalar);
    }

    Matrix<Scalar_, Dynamic, Dynamic> operator-(Scalar_ scalar) const {
        return Matrix<Scalar_, Dynamic, Dynamic>(m_tensor - scalar);
    }

    Matrix<Scalar_, Dynamic, Dynamic> operator-() const {
        return Matrix<Scalar_, Dynamic, Dynamic>(-m_tensor);
    }

    friend Matrix<Scalar_, Dynamic, Dynamic> operator*(Scalar_ scalar, const Matrix& m) {
        return Matrix<Scalar_, Dynamic, Dynamic>(scalar * m.tensor());
    }

    Matrix& operator+=(const Matrix& other) {
        m_tensor = m_tensor + other.tensor();
        return *this;
    }

    Matrix& operator-=(const Matrix& other) {
        m_tensor = m_tensor - other.tensor();
        return *this;
    }

    Matrix& operator*=(Scalar_ scalar) {
        m_tensor = m_tensor * scalar;
        return *this;
    }

    Matrix& operator/=(Scalar_ scalar) {
        m_tensor = m_tensor / scalar;
        return *this;
    }

    Matrix& operator*=(const Matrix& other) {
        m_tensor = torch::matmul(m_tensor, other.tensor());
        return *this;
    }

    // Transpose
    Matrix<Scalar_, Dynamic, Dynamic> transpose() const {
        return Matrix<Scalar_, Dynamic, Dynamic>(m_tensor.transpose(0, 1));
    }

    void transposeInPlace() {
        m_tensor = m_tensor.transpose(0, 1).clone();
    }

    // Adjoint (conjugate transpose)
    Matrix<Scalar_, Dynamic, Dynamic> adjoint() const {
        // Since we primarily deal with real numbers for now, adjoint == transpose. If complex, would need conjunct.
        // For PyTorch: at::conj(m_tensor).transpose(0, 1) if complex, but adjoint() in Torch does that natively.
        return Matrix<Scalar_, Dynamic, Dynamic>(m_tensor.adjoint());
    }

    void adjointInPlace() {
        m_tensor = m_tensor.adjoint().clone();
    }

    // Dot product
    template <typename OtherScalar, int OtherRows, int OtherCols>
    Scalar_ dot(const Matrix<OtherScalar, OtherRows, OtherCols>& other) const {
        // In Eigen, dot is generally for vectors. We will flatten and dot to match behavior or just use torch::dot 
        // Note: torch::dot only supports 1D tensors.
        return torch::dot(m_tensor.reshape({-1}), other.tensor().reshape({-1})).template item<Scalar_>();
    }

    // 3D cross product compatibility helper.
    template <typename OtherScalar, int OtherRows, int OtherCols>
    Matrix<Scalar_, 3, 1> cross(const Matrix<OtherScalar, OtherRows, OtherCols>& other) const {
        at::Tensor lhs = m_tensor.reshape({-1});
        at::Tensor rhs = other.tensor().reshape({-1});
        if (lhs.numel() < 3 || rhs.numel() < 3) {
            throw std::invalid_argument("cross() requires vectors of at least size 3");
        }
        at::Tensor result = torch::cross(lhs.slice(0, 0, 3), rhs.slice(0, 0, 3), 0).unsqueeze(1);
        return Matrix<Scalar_, 3, 1>(result);
    }

    // 4D cross3 compatibility helper using xyz coefficients.
    template <typename OtherScalar, int OtherRows, int OtherCols>
    Matrix<Scalar_, 3, 1> cross3(const Matrix<OtherScalar, OtherRows, OtherCols>& other) const {
        at::Tensor lhs = m_tensor.reshape({-1});
        at::Tensor rhs = other.tensor().reshape({-1});
        if (lhs.numel() < 4 || rhs.numel() < 4) {
            throw std::invalid_argument("cross3() requires vectors of at least size 4");
        }
        at::Tensor result = torch::cross(lhs.slice(0, 0, 3), rhs.slice(0, 0, 3), 0).unsqueeze(1);
        return Matrix<Scalar_, 3, 1>(result);
    }

    Matrix<Scalar_, Dynamic, 1> homogeneous() const {
        at::Tensor flat = m_tensor.reshape({-1});
        at::Tensor one = torch::ones({1}, m_tensor.options());
        return Matrix<Scalar_, Dynamic, 1>(torch::cat({flat, one}, 0).unsqueeze(1));
    }

    Matrix<Scalar_, Dynamic, 1> hnormalized() const {
        at::Tensor flat = m_tensor.reshape({-1});
        if (flat.numel() < 2) {
            throw std::invalid_argument("hnormalized() requires a vector of at least size 2");
        }
        at::Tensor last = flat.slice(0, flat.numel() - 1, flat.numel());
        at::Tensor numer = flat.slice(0, 0, flat.numel() - 1);
        return Matrix<Scalar_, Dynamic, 1>((numer / last).unsqueeze(1));
    }

    Matrix<Scalar_, Dynamic, 1> unitOrthogonal() const {
        at::Tensor v = m_tensor.reshape({-1});
        if (v.numel() < 2) {
            throw std::invalid_argument("unitOrthogonal() requires a vector of size >= 2");
        }

        if (v.numel() == 2) {
            at::Tensor ortho = torch::stack({-v[1], v[0]});
            ortho = ortho / torch::linalg_vector_norm(ortho);
            return Matrix<Scalar_, Dynamic, 1>(ortho.unsqueeze(1));
        }

        at::Tensor abs_v = torch::abs(v);
        int64_t min_idx = abs_v.argmin().item<int64_t>();
        at::Tensor basis = torch::zeros_like(v);
        basis[min_idx] = static_cast<Scalar_>(1);

        at::Tensor vv = torch::dot(v, v);
        double vv_abs = torch::abs(vv).template item<double>();
        if (vv_abs <= 1e-12) {
            at::Tensor fallback = torch::zeros_like(v);
            fallback[0] = static_cast<Scalar_>(1);
            return Matrix<Scalar_, Dynamic, 1>(fallback.unsqueeze(1));
        }

        at::Tensor proj = basis - v * (torch::dot(v, basis) / vv);
        proj = proj / torch::linalg_vector_norm(proj);
        return Matrix<Scalar_, Dynamic, 1>(proj.unsqueeze(1));
    }

    // Block operations (creates views)
    Matrix<Scalar_, Dynamic, Dynamic> block(int startRow, int startCol, int blockRows, int blockCols) {
        return Matrix<Scalar_, Dynamic, Dynamic>(
            m_tensor.slice(0, startRow, startRow + blockRows)
                    .slice(1, startCol, startCol + blockCols)
        );
    }
    
    // Const block operations
    const Matrix<Scalar_, Dynamic, Dynamic> block(int startRow, int startCol, int blockRows, int blockCols) const {
        return Matrix<Scalar_, Dynamic, Dynamic>(
            m_tensor.slice(0, startRow, startRow + blockRows)
                    .slice(1, startCol, startCol + blockCols)
        );
    }

    Matrix<Scalar_, Dynamic, Dynamic> topLeftCorner(int blockRows, int blockCols) {
        return block(0, 0, blockRows, blockCols);
    }

    Matrix<Scalar_, Dynamic, Dynamic> topRightCorner(int blockRows, int blockCols) {
        return block(0, cols() - blockCols, blockRows, blockCols);
    }

    Matrix<Scalar_, Dynamic, Dynamic> bottomLeftCorner(int blockRows, int blockCols) {
        return block(rows() - blockRows, 0, blockRows, blockCols);
    }

    Matrix<Scalar_, Dynamic, Dynamic> bottomRightCorner(int blockRows, int blockCols) {
        return block(rows() - blockRows, cols() - blockCols, blockRows, blockCols);
    }

    Matrix<Scalar_, Dynamic, Dynamic> topRows(int blockRows) {
        return block(0, 0, blockRows, cols());
    }

    Matrix<Scalar_, Dynamic, Dynamic> bottomRows(int blockRows) {
        return block(rows() - blockRows, 0, blockRows, cols());
    }

    Matrix<Scalar_, Dynamic, Dynamic> leftCols(int blockCols) {
        return block(0, 0, rows(), blockCols);
    }

    Matrix<Scalar_, Dynamic, Dynamic> rightCols(int blockCols) {
        return block(0, cols() - blockCols, rows(), blockCols);
    }

    Matrix<Scalar_, Dynamic, 1> head(int n) const {
        return Matrix<Scalar_, Dynamic, 1>(m_tensor.reshape({-1}).slice(0, 0, n).unsqueeze(1));
    }

    Matrix<Scalar_, Dynamic, 1> tail(int n) const {
        int total = size();
        return Matrix<Scalar_, Dynamic, 1>(m_tensor.reshape({-1}).slice(0, total - n, total).unsqueeze(1));
    }

    Matrix<Scalar_, Dynamic, 1> segment(int start, int n) const {
        return Matrix<Scalar_, Dynamic, 1>(m_tensor.reshape({-1}).slice(0, start, start + n).unsqueeze(1));
    }

    // Row extraction
    Matrix<Scalar_, 1, Dynamic> row(int i) {
        return Matrix<Scalar_, 1, Dynamic>(m_tensor.slice(0, i, i + 1));
    }

    const Matrix<Scalar_, 1, Dynamic> row(int i) const {
        return Matrix<Scalar_, 1, Dynamic>(m_tensor.slice(0, i, i + 1));
    }

    // Column extraction
    Matrix<Scalar_, Dynamic, 1> col(int i) {
        return Matrix<Scalar_, Dynamic, 1>(m_tensor.slice(1, i, i + 1));
    }

    const Matrix<Scalar_, Dynamic, 1> col(int i) const {
        return Matrix<Scalar_, Dynamic, 1>(m_tensor.slice(1, i, i + 1));
    }

    // Element-wise operations
    Matrix<Scalar_, Dynamic, Dynamic> cwiseProduct(const Matrix& other) const {
        return Matrix<Scalar_, Dynamic, Dynamic>(m_tensor * other.tensor());
    }

    Matrix<Scalar_, Dynamic, Dynamic> cwiseQuotient(const Matrix& other) const {
        return Matrix<Scalar_, Dynamic, Dynamic>(m_tensor / other.tensor());
    }

    Matrix<Scalar_, Dynamic, Dynamic> cwiseAbs() const {
        return Matrix<Scalar_, Dynamic, Dynamic>(torch::abs(m_tensor));
    }

    Matrix<Scalar_, Dynamic, Dynamic> cwiseAbs2() const {
        return Matrix<Scalar_, Dynamic, Dynamic>(torch::abs(m_tensor) * torch::abs(m_tensor));
    }

    Matrix<Scalar_, Dynamic, Dynamic> cwiseSqrt() const {
        return Matrix<Scalar_, Dynamic, Dynamic>(torch::sqrt(m_tensor));
    }

    Matrix<Scalar_, Dynamic, Dynamic> cwiseInverse() const {
        return Matrix<Scalar_, Dynamic, Dynamic>(torch::reciprocal(m_tensor));
    }

    Matrix<Scalar_, Dynamic, Dynamic> cwiseMin(const Matrix& other) const {
        return Matrix<Scalar_, Dynamic, Dynamic>(torch::minimum(m_tensor, other.tensor()));
    }

    Matrix<Scalar_, Dynamic, Dynamic> cwiseMax(const Matrix& other) const {
        return Matrix<Scalar_, Dynamic, Dynamic>(torch::maximum(m_tensor, other.tensor()));
    }

    Matrix<Scalar_, Dynamic, Dynamic> cwiseMin(Scalar_ scalar) const {
        return Matrix<Scalar_, Dynamic, Dynamic>(torch::minimum(m_tensor, torch::full_like(m_tensor, scalar)));
    }

    Matrix<Scalar_, Dynamic, Dynamic> cwiseMax(Scalar_ scalar) const {
        return Matrix<Scalar_, Dynamic, Dynamic>(torch::maximum(m_tensor, torch::full_like(m_tensor, scalar)));
    }

    Matrix<Scalar_, Dynamic, Dynamic> conjugate() const {
        return Matrix<Scalar_, Dynamic, Dynamic>(torch::conj(m_tensor));
    }

    Matrix<Scalar_, Dynamic, Dynamic> real() const {
        return Matrix<Scalar_, Dynamic, Dynamic>(torch::real(m_tensor));
    }

    Matrix<Scalar_, Dynamic, Dynamic> imag() const {
        return Matrix<Scalar_, Dynamic, Dynamic>(torch::imag(m_tensor));
    }

    // Reductions
    Scalar_ sum() const {
        return m_tensor.sum().template item<Scalar_>();
    }

    Scalar_ prod() const {
        return m_tensor.prod().template item<Scalar_>();
    }

    Scalar_ mean() const {
        return m_tensor.mean().template item<Scalar_>();
    }

    Scalar_ maxCoeff() const {
        return m_tensor.max().template item<Scalar_>();
    }

    Scalar_ minCoeff() const {
        return m_tensor.min().template item<Scalar_>();
    }

    Scalar_ squaredNorm() const {
        at::Tensor flat = m_tensor.reshape({-1});
        return torch::real(torch::dot(torch::conj(flat), flat)).template item<Scalar_>();
    }

    Scalar_ norm() const {
        return torch::linalg_vector_norm(m_tensor.reshape({-1})).template item<Scalar_>();
    }

    Matrix normalized() const {
        Scalar_ n = norm();
        if (n == static_cast<Scalar_>(0)) {
            return Matrix(m_tensor.clone());
        }
        return Matrix(m_tensor / n);
    }

    Matrix& normalize() {
        Scalar_ n = norm();
        if (n != static_cast<Scalar_>(0)) {
            m_tensor = m_tensor / n;
        }
        return *this;
    }

    // Linear Algebra
    Matrix<Scalar_, Dynamic, Dynamic> inverse() const {
        // Note: torch::inverse expects float or double
        return Matrix<Scalar_, Dynamic, Dynamic>(torch::inverse(m_tensor));
    }

    Scalar_ determinant() const {
        return torch::linalg_det(m_tensor).template item<Scalar_>();
    }

    Matrix eval() const {
        return *this;
    }

    Array<Scalar_, RowsAtCompileTime_, ColsAtCompileTime_> array();
    const Array<Scalar_, RowsAtCompileTime_, ColsAtCompileTime_> array() const;

    Matrix& matrix() { return *this; }
    const Matrix& matrix() const { return *this; }

    template <typename UnaryFunc>
    Matrix unaryExpr(UnaryFunc fn) const {
        at::Tensor cpu = m_tensor.to(torch::kCPU).contiguous();
        at::Tensor out = torch::empty_like(cpu);

        Scalar_* out_ptr = out.data_ptr<Scalar_>();
        const Scalar_* in_ptr = cpu.data_ptr<Scalar_>();
        for (int i = 0; i < cpu.numel(); ++i) {
            out_ptr[i] = fn(in_ptr[i]);
        }

        return Matrix(out.to(m_tensor.device()));
    }

    // Eigen-style decomposition chaining (available when including <Eigen/Dense>)
    PartialPivLU<Matrix> partialPivLu() const;
    FullPivLU<Matrix> fullPivLu() const;
    HouseholderQR<Matrix> householderQr() const;
    ColPivHouseholderQR<Matrix> colPivHouseholderQr() const;
    FullPivHouseholderQR<Matrix> fullPivHouseholderQr() const;
    LLT<Matrix> llt() const;
    LDLT<Matrix> ldlt() const;
    JacobiSVD<Matrix> jacobiSvd(unsigned int computationOptions = 0) const;
    BDCSVD<Matrix> bdcSvd(unsigned int computationOptions = 0) const;
    CompleteOrthogonalDecomposition<Matrix> completeOrthogonalDecomposition() const;
    EigenSolver<Matrix> eigenSolver(bool computeEigenvectors = true) const;
    SelfAdjointEigenSolver<Matrix> selfAdjointEigenSolver(bool computeEigenvectors = true) const;
    ComplexEigenSolver<Matrix> complexEigenSolver(bool computeEigenvectors = true) const;

    // Stream operator for printing
    friend std::ostream& operator<<(std::ostream& os, const Matrix& m) {
        os << m.tensor();
        return os;
    }
};

// Common dynamic typedefs
using MatrixXf = Matrix<float, Dynamic, Dynamic>;
using MatrixXd = Matrix<double, Dynamic, Dynamic>;
using MatrixXcf = Matrix<std::complex<float>, Dynamic, Dynamic>;
using MatrixXcd = Matrix<std::complex<double>, Dynamic, Dynamic>;
using MatrixXi = Matrix<int32_t, Dynamic, Dynamic>;
using VectorXf = Matrix<float, Dynamic, 1>;
using VectorXd = Matrix<double, Dynamic, 1>;
using VectorXcf = Matrix<std::complex<float>, Dynamic, 1>;
using VectorXcd = Matrix<std::complex<double>, Dynamic, 1>;
using VectorXi = Matrix<int32_t, Dynamic, 1>;
using RowVectorXf = Matrix<float, 1, Dynamic>;
using RowVectorXd = Matrix<double, 1, Dynamic>;

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

using RowVector2f = Matrix<float, 1, 2>;
using RowVector3f = Matrix<float, 1, 3>;
using RowVector4f = Matrix<float, 1, 4>;

using RowVector2d = Matrix<double, 1, 2>;
using RowVector3d = Matrix<double, 1, 3>;
using RowVector4d = Matrix<double, 1, 4>;

} // namespace Eigen
