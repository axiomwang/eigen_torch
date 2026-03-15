#include <Eigen/Sparse>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

float maxAbs(const Eigen::MatrixXf& m) {
    return m.cwiseAbs().maxCoeff();
}

bool matrixNear(const Eigen::MatrixXf& lhs, const Eigen::MatrixXf& rhs, float eps = 1e-5f) {
    return maxAbs(lhs - rhs) <= eps;
}

bool near(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) <= eps;
}

bool isUnitLowerTriangular(const Eigen::MatrixXf& m, float eps = 1e-5f) {
    for (int row = 0; row < m.rows(); ++row) {
        for (int col = 0; col < m.cols(); ++col) {
            if (row < col && std::abs(m(row, col)) > eps) {
                return false;
            }
            if (row == col && std::abs(m(row, col) - 1.0f) > eps) {
                return false;
            }
        }
    }
    return true;
}

bool isUpperTriangular(const Eigen::MatrixXf& m, float eps = 1e-5f) {
    for (int row = 0; row < m.rows(); ++row) {
        for (int col = 0; col < m.cols(); ++col) {
            if (row > col && std::abs(m(row, col)) > eps) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main() {
    Eigen::DeviceManager::set_default_device(torch::kCPU);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    std::vector<Eigen::Triplet<float>> triplets = {
        {0, 0, 4.0f}, {0, 1, 1.0f}, {0, 1, 0.5f},
        {1, 0, 1.5f}, {1, 1, 4.0f}, {1, 2, 1.0f},
        {2, 1, 1.0f}, {2, 2, 4.0f}, {2, 3, 1.0f},
        {3, 2, 1.0f}, {3, 3, 3.0f}, {3, 3, 0.0f}
    };

    Eigen::SparseMatrixf A(4, 4);
    A.setFromTriplets(triplets.begin(), triplets.end());

    if (!A.isCompressed()) {
        std::cerr << "SparseMatrix should be compressed after setFromTriplets" << std::endl;
        return 1;
    }

    if (!near(A.coeff(0, 1), 1.5f)) {
        std::cerr << "Triplet duplicate merge failed for coeff(0,1)" << std::endl;
        return 1;
    }

    A.coeffRef(0, 2) = 0.25f;
    A.coeffRef(2, 0) = 0.25f;
    A.makeCompressed();

    Eigen::MatrixXf expected_dense(torch::tensor({
        {4.0f, 1.5f, 0.25f, 0.0f},
        {1.5f, 4.0f, 1.0f, 0.0f},
        {0.25f, 1.0f, 4.0f, 1.0f},
        {0.0f, 0.0f, 1.0f, 3.0f}
    }, options));

    if (A.nonZeros() != 12 || !near(A.coeff(0, 2), 0.25f) || !matrixNear(A.toDense(), expected_dense)) {
        std::cerr << "SparseMatrix dense materialization mismatch" << std::endl;
        return 1;
    }

    int iterCount = 0;
    for (int outer = 0; outer < A.outerSize(); ++outer) {
        for (Eigen::SparseMatrixf::InnerIterator it(A, outer); it; ++it) {
            ++iterCount;
        }
    }
    if (iterCount != A.nonZeros()) {
        std::cerr << "InnerIterator traversal count mismatch" << std::endl;
        return 1;
    }

    Eigen::SparseVectorf sv(4);
    sv.insert(0) = 1.0f;
    sv.coeffRef(3) = 2.0f;
    sv.makeCompressed();
    Eigen::VectorXf expected_sv(torch::tensor({1.0f, 0.0f, 0.0f, 2.0f}, options).unsqueeze(1));
    if (sv.nonZeros() != 2 || !near(sv.coeff(3), 2.0f) || !matrixNear(sv.toDense(), expected_sv)) {
        std::cerr << "SparseVector mismatch" << std::endl;
        return 1;
    }

    Eigen::MatrixXf x_true = Eigen::MatrixXf::Ones(4, 1);
    Eigen::MatrixXf b = A * x_true;
    Eigen::VectorXf expected_b(torch::tensor({5.75f, 6.5f, 6.25f, 4.0f}, options).unsqueeze(1));
    Eigen::VectorXf expected_sv_product(torch::tensor({4.0f, 1.5f, 2.25f, 6.0f}, options).unsqueeze(1));
    if (!matrixNear(b, expected_b) || !matrixNear(A * sv.toDense(), expected_sv_product)) {
        std::cerr << "Sparse multiply mismatch" << std::endl;
        return 1;
    }

    Eigen::SparseMatrixf identity(4, 4);
    identity.setIdentity();
    if (!matrixNear((A * identity).toDense(), expected_dense)) {
        std::cerr << "Sparse-sparse multiply mismatch" << std::endl;
        return 1;
    }

    Eigen::SparseLU<Eigen::SparseMatrixf> lu;
    lu.analyzePattern(A);
    lu.factorize(A);
    if (lu.info() != Eigen::Success) {
        std::cerr << "SparseLU factorization failed" << std::endl;
        return 1;
    }
    Eigen::MatrixXf x_lu = lu.solve(b);
    if (maxAbs(A.toDense() * x_lu - b) > 1e-3f) {
        std::cerr << "SparseLU solve residual too large" << std::endl;
        return 1;
    }
    Eigen::MatrixXf lu_l = lu.matrixL();
    Eigen::MatrixXf lu_u = lu.matrixU();
    if (!matrixNear(x_lu, x_true, 1e-3f) || !isUnitLowerTriangular(lu_l) || !isUpperTriangular(lu_u)) {
        std::cerr << "SparseLU factor structure mismatch" << std::endl;
        return 1;
    }

    Eigen::SparseQR<Eigen::SparseMatrixf> qr;
    qr.compute(A);
    if (qr.info() != Eigen::Success) {
        std::cerr << "SparseQR factorization failed" << std::endl;
        return 1;
    }
    Eigen::MatrixXf x_qr = qr.solve(b);
    if (maxAbs(A.toDense() * x_qr - b) > 1e-3f) {
        std::cerr << "SparseQR solve residual too large" << std::endl;
        return 1;
    }
    Eigen::MatrixXf qr_q = qr.matrixQ();
    Eigen::MatrixXf qr_r = qr.matrixR();
    if (!matrixNear(x_qr, x_true, 1e-3f) ||
        !matrixNear(qr_q.transpose() * qr_q, Eigen::MatrixXf::Identity(4, 4), 1e-3f) ||
        !matrixNear(qr_q * qr_r, expected_dense, 1e-3f)) {
        std::cerr << "SparseQR reconstruction mismatch" << std::endl;
        return 1;
    }

    Eigen::SimplicialLLT<Eigen::SparseMatrixf> llt;
    llt.analyzePattern(A);
    llt.factorize(A);
    if (llt.info() != Eigen::Success) {
        std::cerr << "SimplicialLLT factorization failed" << std::endl;
        return 1;
    }
    Eigen::MatrixXf x_llt = llt.solve(b);
    if (maxAbs(A.toDense() * x_llt - b) > 1e-3f) {
        std::cerr << "SimplicialLLT solve residual too large" << std::endl;
        return 1;
    }
    Eigen::MatrixXf llt_l = llt.matrixL();
    Eigen::MatrixXf llt_u = llt.matrixU();
    if (!matrixNear(x_llt, x_true, 1e-3f) || !matrixNear(llt_l * llt_u, expected_dense, 1e-3f)) {
        std::cerr << "SimplicialLLT reconstruction mismatch" << std::endl;
        return 1;
    }

    Eigen::SimplicialLDLT<Eigen::SparseMatrixf> ldlt;
    ldlt.compute(A);
    if (ldlt.info() != Eigen::Success) {
        std::cerr << "SimplicialLDLT factorization failed" << std::endl;
        return 1;
    }
    Eigen::MatrixXf x_ldlt = ldlt.solve(b);
    if (maxAbs(A.toDense() * x_ldlt - b) > 1e-3f) {
        std::cerr << "SimplicialLDLT solve residual too large" << std::endl;
        return 1;
    }
    Eigen::MatrixXf ldlt_l = ldlt.matrixL();
    Eigen::MatrixXf ldlt_u = ldlt.matrixU();
    Eigen::VectorXf ldlt_d = ldlt.vectorD();
    Eigen::MatrixXf d_matrix = Eigen::MatrixXf::Zero(4, 4);
    for (int index = 0; index < 4; ++index) {
        d_matrix(index, index) = ldlt_d(index);
    }
    if (!matrixNear(x_ldlt, x_true, 1e-3f) || !matrixNear(ldlt_l * d_matrix * ldlt_u, expected_dense, 1e-3f)) {
        std::cerr << "SimplicialLDLT reconstruction mismatch" << std::endl;
        return 1;
    }

    Eigen::SparseMatrix<float, Eigen::RowMajor> A_row(A.tensor());
    Eigen::MatrixXf b_row = A_row * x_true;
    if (maxAbs(b_row - b) > 1e-5f) {
        std::cerr << "RowMajor sparse multiply mismatch" << std::endl;
        return 1;
    }

    std::cout << "Sparse API/linalg checks passed" << std::endl;
    return 0;
}
