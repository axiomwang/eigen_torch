#include <Eigen/Sparse>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

float maxAbs(const Eigen::MatrixXf& m) {
    return m.cwiseAbs().maxCoeff();
}

bool near(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) <= eps;
}

}  // namespace

int main() {
    Eigen::DeviceManager::set_default_device(torch::kCPU);

    // Build an SPD sparse matrix with duplicate triplets to test merge behavior.
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

    A.coeffRef(1, 2) = 0.25f;
    A.coeffRef(2, 1) = 0.25f;
    A.makeCompressed();

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

    Eigen::MatrixXf x_true = Eigen::MatrixXf::Ones(4, 1);
    Eigen::MatrixXf b = A * x_true;

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
    (void)lu.matrixL();
    (void)lu.matrixU();

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
    (void)qr.matrixQ();
    (void)qr.matrixR();

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
    (void)llt.matrixL();
    (void)llt.matrixU();

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
    (void)ldlt.matrixL();
    (void)ldlt.matrixU();
    (void)ldlt.vectorD();

    Eigen::SparseMatrix<float, Eigen::RowMajor> A_row(A.tensor());
    Eigen::MatrixXf b_row = A_row * x_true;
    if (maxAbs(b_row - b) > 1e-5f) {
        std::cerr << "RowMajor sparse multiply mismatch" << std::endl;
        return 1;
    }

    std::cout << "Sparse API/linalg checks passed" << std::endl;
    return 0;
}
