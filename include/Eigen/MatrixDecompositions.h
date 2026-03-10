#pragma once

#include "Matrix.h"
#include "LU"
#include "QR"

namespace Eigen {

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
inline FullPivLU<Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>>
Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>::fullPivLu() const {
    using Self = Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>;
    return FullPivLU<Self>(static_cast<const Self&>(*this));
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

} // namespace Eigen

