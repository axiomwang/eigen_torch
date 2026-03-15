#pragma once

#include "Matrix.h"
#include "LU"
#include "Cholesky"
#include "QR"
#include "SVD"
#include "Eigenvalues"

namespace Eigen {

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
inline PartialPivLU<Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>>
Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>::partialPivLu() const {
    using Self = Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>;
    return PartialPivLU<Self>(static_cast<const Self&>(*this));
}

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
inline FullPivLU<Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>>
Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>::fullPivLu() const {
    using Self = Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>;
    return FullPivLU<Self>(static_cast<const Self&>(*this));
}

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
inline HouseholderQR<Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>>
Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>::householderQr() const {
    using Self = Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>;
    return HouseholderQR<Self>(static_cast<const Self&>(*this));
}

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
inline ColPivHouseholderQR<Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>>
Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>::colPivHouseholderQr() const {
    using Self = Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>;
    return ColPivHouseholderQR<Self>(static_cast<const Self&>(*this));
}

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
inline FullPivHouseholderQR<Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>>
Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>::fullPivHouseholderQr() const {
    using Self = Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>;
    return FullPivHouseholderQR<Self>(static_cast<const Self&>(*this));
}

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
inline LLT<Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>>
Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>::llt() const {
    using Self = Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>;
    return LLT<Self>(static_cast<const Self&>(*this));
}

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
inline LDLT<Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>>
Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>::ldlt() const {
    using Self = Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>;
    return LDLT<Self>(static_cast<const Self&>(*this));
}

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
inline JacobiSVD<Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>>
Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>::jacobiSvd(unsigned int computationOptions) const {
    using Self = Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>;
    return JacobiSVD<Self>(static_cast<const Self&>(*this), computationOptions);
}

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
inline BDCSVD<Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>>
Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>::bdcSvd(unsigned int computationOptions) const {
    using Self = Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>;
    return BDCSVD<Self>(static_cast<const Self&>(*this), computationOptions);
}

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
inline CompleteOrthogonalDecomposition<Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>>
Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>::completeOrthogonalDecomposition() const {
    using Self = Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>;
    return CompleteOrthogonalDecomposition<Self>(static_cast<const Self&>(*this));
}

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
inline EigenSolver<Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>>
Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>::eigenSolver(bool computeEigenvectors) const {
    using Self = Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>;
    return EigenSolver<Self>(static_cast<const Self&>(*this), computeEigenvectors);
}

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
inline SelfAdjointEigenSolver<Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>>
Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>::selfAdjointEigenSolver(bool computeEigenvectors) const {
    using Self = Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>;
    return SelfAdjointEigenSolver<Self>(static_cast<const Self&>(*this), computeEigenvectors);
}

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
inline ComplexEigenSolver<Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>>
Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>::complexEigenSolver(bool computeEigenvectors) const {
    using Self = Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>;
    return ComplexEigenSolver<Self>(static_cast<const Self&>(*this), computeEigenvectors);
}

} // namespace Eigen

