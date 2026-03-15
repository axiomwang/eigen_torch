#include <unsupported/Eigen/AdolcForward>
#include <unsupported/Eigen/AlignedVector3>
#include <unsupported/Eigen/ArpackSupport>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/TensorSymmetry>
#include <unsupported/Eigen/CXX11/ThreadPool>
#include <unsupported/Eigen/IterativeSolvers>
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/LevenbergMarquardt>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/MPRealSupport>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <unsupported/Eigen/OpenGLSupport>
#include <unsupported/Eigen/Polynomials>
#include <unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/SpecialFunctions>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

namespace {

template <typename T, typename U>
bool near(T lhs, U rhs, double eps = 1e-4) {
    return std::abs(static_cast<double>(lhs) - static_cast<double>(rhs)) <= eps;
}

template <typename Scalar, int RowsA, int ColsA, int RowsB, int ColsB>
bool matrixNear(
    const Eigen::Matrix<Scalar, RowsA, ColsA>& lhs,
    const Eigen::Matrix<Scalar, RowsB, ColsB>& rhs,
    double eps = 1e-4) {
    return (lhs - rhs).cwiseAbs().maxCoeff() <= static_cast<Scalar>(eps);
}

struct QuadraticResidualFunctor {
    Eigen::VectorXf operator()(const Eigen::VectorXf& x) const {
        at::Tensor x_flat = x.tensor().reshape({-1});
        at::Tensor residual = torch::stack({
            x_flat.index({0}) * x_flat.index({0}) + x_flat.index({1}),
            x_flat.index({0}) + x_flat.index({1}) * x_flat.index({1})
        }).reshape({2, 1});
        return Eigen::VectorXf(residual);
    }
};

struct TargetResidualFunctor {
    Eigen::VectorXf operator()(const Eigen::VectorXf& x) const {
        at::Tensor x_flat = x.tensor().reshape({-1});
        at::Tensor residual = torch::stack({
            x_flat.index({0}) - 3.0f,
            x_flat.index({1}) + 2.0f
        }).reshape({2, 1});
        return Eigen::VectorXf(residual);
    }
};

} // namespace

int main() {
    Eigen::DeviceManager::set_default_device(torch::kCPU);

    {
        float derivative = Eigen::AdolcForward<float>::derivative(
            [](const at::Tensor& x) {
                return torch::pow(x, 3).sum() + 2.0f * x.sum();
            },
            2.0f);

        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        Eigen::VectorXf x(torch::tensor({1.0f, 2.0f}, options).unsqueeze(1));
        Eigen::VectorXf gradient = Eigen::AdolcForward<float>::gradient(
            [](const at::Tensor& v) {
                return (v * v).sum();
            },
            x);

        if (!near(derivative, 14.0f, 1e-3) || !near(gradient(0, 0), 2.0f, 1e-3) || !near(gradient(1, 0), 4.0f, 1e-3)) {
            std::cerr << "AdolcForward mismatch" << std::endl;
            return 1;
        }
    }

    {
        Eigen::AlignedVector3f a(1.0f, 2.0f, 3.0f);
        Eigen::AlignedVector3f b(-1.0f, 0.0f, 2.0f);
        Eigen::AlignedVector3f cross = a.cross(b);
        Eigen::AlignedVector3f sum = a + b;

        if (!near(a.dot(b), 5.0f, 1e-4) ||
            !near(cross.x(), 4.0f, 1e-4) ||
            !near(cross.y(), -5.0f, 1e-4) ||
            !near(cross.z(), 2.0f, 1e-4) ||
            !near(sum.x(), 0.0f, 1e-4) ||
            !near(a.normalized().norm(), 1.0f, 1e-4)) {
            std::cerr << "AlignedVector3 mismatch" << std::endl;
            return 1;
        }
    }

    {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        Eigen::MatrixXf A(torch::tensor({
            {1.0f, 0.0f, 0.0f},
            {0.0f, 3.0f, 0.0f},
            {0.0f, 0.0f, 2.0f}
        }, options));

        Eigen::ArpackSelfAdjointEigenSolver<Eigen::MatrixXf> solver(A, 2);
        Eigen::VectorXf values = solver.eigenvalues();
        Eigen::MatrixXf vectors = solver.eigenvectors();

        Eigen::MatrixXf D = Eigen::MatrixXf::Zero(2, 2);
        D(0, 0) = values(0);
        D(1, 1) = values(1);
        Eigen::MatrixXf residual = A * vectors - vectors * D;

        if (!solver.info() || !near(values(0), 2.0f, 1e-4) || !near(values(1), 3.0f, 1e-4) || residual.norm() > 1e-4f) {
            std::cerr << "ArpackSupport mismatch" << std::endl;
            return 1;
        }
    }

    {
        Eigen::Tensor<float, 2> t({2, 2});
        t(0, 0) = 1.0f;
        t(0, 1) = 2.0f;
        t(1, 0) = 3.0f;
        t(1, 1) = 4.0f;

        auto sym = Eigen::symmetrize_matrix_tensor(t);
        auto asym = Eigen::antisymmetrize_matrix_tensor(t);

        if (!near(sym(0, 0), 1.0f, 1e-5) || !near(sym(0, 1), 2.5f, 1e-5) ||
            !near(sym(1, 0), 2.5f, 1e-5) || !near(sym(1, 1), 4.0f, 1e-5) ||
            !near(asym(0, 1), -0.5f, 1e-5) || !near(asym(1, 0), 0.5f, 1e-5)) {
            std::cerr << "TensorSymmetry mismatch" << std::endl;
            return 1;
        }

        Eigen::ThreadPoolDevice pool(2);
        std::atomic<int64_t> total{0};
        pool.parallelFor(0, 100, 10, [&](int begin, int end) {
            int64_t partial = 0;
            for (int index = begin; index < end; ++index) {
                partial += index;
            }
            total.fetch_add(partial);
        });

        Eigen::Tensor<float, 2> lhs({2, 2});
        lhs(0, 0) = 1.0f;
        lhs(0, 1) = 2.0f;
        lhs(1, 0) = 3.0f;
        lhs(1, 1) = 4.0f;

        Eigen::Tensor<float, 2> rhs({2, 2});
        rhs(0, 0) = 5.0f;
        rhs(0, 1) = 6.0f;
        rhs(1, 0) = 7.0f;
        rhs(1, 1) = 8.0f;

        Eigen::Tensor<float, 2> product = pool.parallelMatMul(lhs, rhs);

        if (total.load() != 4950 || !near(product(0, 0), 19.0f, 1e-5) || !near(product(1, 1), 50.0f, 1e-5)) {
            std::cerr << "ThreadPool mismatch" << std::endl;
            return 1;
        }
    }

    {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

        Eigen::MatrixXf A(torch::tensor({
            {4.0f, 1.0f},
            {1.0f, 3.0f}
        }, options));
        Eigen::VectorXf b(torch::tensor({1.0f, 2.0f}, options).unsqueeze(1));

        Eigen::ConjugateGradient<Eigen::MatrixXf> cg;
        cg.setMaxIterations(128).setTolerance(1e-8).compute(A);
        Eigen::VectorXf x = cg.solve(b);

        if (cg.info() != Eigen::Success || !near(x(0), 1.0f / 11.0f, 1e-3) || !near(x(1), 7.0f / 11.0f, 1e-3)) {
            std::cerr << "ConjugateGradient mismatch" << std::endl;
            return 1;
        }

        Eigen::MatrixXf B(torch::tensor({
            {4.0f, 2.0f},
            {1.0f, 3.0f}
        }, options));

        Eigen::BiCGSTAB<Eigen::MatrixXf> bicg;
        bicg.setMaxIterations(256).setTolerance(1e-8).compute(B);
        Eigen::VectorXf x_bicg = bicg.solve(b);

        if (bicg.info() != Eigen::Success || !near(x_bicg(0), -0.1f, 1e-3) || !near(x_bicg(1), 0.7f, 1e-3)) {
            std::cerr << "BiCGSTAB mismatch" << std::endl;
            return 1;
        }
    }

    {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        Eigen::MatrixXf lhs(torch::tensor({
            {1.0f, 2.0f},
            {3.0f, 4.0f}
        }, options));
        Eigen::MatrixXf rhs(torch::tensor({
            {0.0f, 5.0f},
            {6.0f, 7.0f}
        }, options));

        Eigen::MatrixXf kron = Eigen::kroneckerProduct(lhs, rhs);
        Eigen::MatrixXf expected(torch::tensor({
            {0.0f, 5.0f, 0.0f, 10.0f},
            {6.0f, 7.0f, 12.0f, 14.0f},
            {0.0f, 15.0f, 0.0f, 20.0f},
            {18.0f, 21.0f, 24.0f, 28.0f}
        }, options));
        Eigen::KroneckerProduct<Eigen::MatrixXf, Eigen::MatrixXf> expr(lhs, rhs);

        if (!matrixNear(kron, expected, 1e-5) || !matrixNear(expr.eval(), expected, 1e-5)) {
            std::cerr << "KroneckerProduct mismatch" << std::endl;
            return 1;
        }
    }

    {
        Eigen::MatrixXf M = Eigen::MatrixXf::Zero(2, 2);
        M(0, 0) = 1.0f;
        M(1, 1) = 2.0f;

        auto expM = Eigen::matrixExp(M);
        auto logM = Eigen::matrixLog(M);
        auto sqrtM = Eigen::matrixSqrt(M);
        auto powM = Eigen::matrixPower(M, 2);
        auto fracM = Eigen::matrixPow(M, 0.5);
        auto exprM = Eigen::matrixFunctionExp(M).eval();

        if (!near(expM(0, 0), std::exp(1.0f), 1e-3) ||
            !near(expM(1, 1), std::exp(2.0f), 1e-3) ||
            !near(logM(0, 0), 0.0f, 1e-3) ||
            !near(logM(1, 1), std::log(2.0f), 1e-3) ||
            !near(sqrtM(1, 1), std::sqrt(2.0f), 1e-3) ||
            !near(powM(1, 1), 4.0f, 1e-3) ||
            !near(fracM(1, 1), std::sqrt(2.0f), 1e-3) ||
            !matrixNear(expM, exprM, 1e-4)) {
            std::cerr << "MatrixFunctions mismatch" << std::endl;
            return 1;
        }
    }

    {
        using MPS = Eigen::MPRealSupport<double>;
        if (!near(MPS::pi(), std::acos(-1.0), 1e-10) ||
            !near(MPS::sqrt(9.0), 3.0, 1e-10) ||
            !near(MPS::exp(1.0), std::exp(1.0), 1e-10) ||
            !near(MPS::log(std::exp(1.0)), 1.0, 1e-10)) {
            std::cerr << "MPRealSupport scalar mismatch" << std::endl;
            return 1;
        }

        auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
        Eigen::Matrix<double, Eigen::Dynamic, 1> a(torch::tensor({1.0, 2.0, 3.0}, options).unsqueeze(1));
        Eigen::Matrix<double, Eigen::Dynamic, 1> b(torch::tensor({4.0, 5.0, 6.0}, options).unsqueeze(1));
        double dot = MPS::dot(a, b);

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> lhs(torch::tensor({
            {1.0, 2.0},
            {3.0, 4.0}
        }, options));
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> rhs(torch::tensor({
            {5.0, 6.0},
            {7.0, 8.0}
        }, options));
        auto product = MPS::matmul(lhs, rhs);

        if (!near(dot, 32.0, 1e-10) || !near(product(0, 0), 19.0, 1e-10) || !near(product(1, 1), 50.0, 1e-10)) {
            std::cerr << "MPRealSupport tensor mismatch" << std::endl;
            return 1;
        }
    }

    {
        QuadraticResidualFunctor quadratic;
        Eigen::NumericalDiff<QuadraticResidualFunctor> numerical(quadratic, 1e-4);
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        Eigen::VectorXf x(torch::tensor({1.0f, 2.0f}, options).unsqueeze(1));

        Eigen::MatrixXf J = numerical.jacobian(x);
        Eigen::MatrixXf expected(torch::tensor({
            {2.0f, 1.0f},
            {1.0f, 4.0f}
        }, options));

        if (!matrixNear(J, expected, 1e-2)) {
            std::cerr << "NumericalDiff mismatch" << std::endl;
            std::cerr << "J = ["
                      << J(0, 0) << ", " << J(0, 1) << "; "
                      << J(1, 0) << ", " << J(1, 1) << "]" << std::endl;
            return 1;
        }

        TargetResidualFunctor target;

        Eigen::VectorXf x_hybrid(torch::zeros({2, 1}, options));
        Eigen::HybridNonLinearSolver<TargetResidualFunctor> hybrid(target);
        auto hybrid_status = hybrid.setMaxIterations(64).setTolerance(1e-6).setDamping(1e-5).minimize(x_hybrid);

        if (hybrid_status != Eigen::HybridSuccess || !near(x_hybrid(0, 0), 3.0f, 1e-3) || !near(x_hybrid(1, 0), -2.0f, 1e-3)) {
            std::cerr << "NonLinearOptimization mismatch" << std::endl;
            return 1;
        }

        Eigen::VectorXf x_lm(torch::zeros({2, 1}, options));
        Eigen::LevenbergMarquardt<TargetResidualFunctor> lm(target);
        auto lm_status = lm.setMaxIterations(64).setTolerance(1e-6).setLambda(1e-3).minimize(x_lm);

        if (lm_status != Eigen::LMSuccess || !near(x_lm(0, 0), 3.0f, 1e-3) || !near(x_lm(1, 0), -2.0f, 1e-3)) {
            std::cerr << "LevenbergMarquardt mismatch" << std::endl;
            return 1;
        }
    }

    {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

        Eigen::Matrix<float, 4, 4> model(torch::tensor({
            {1.0f, 0.0f, 0.0f, 1.0f},
            {0.0f, 1.0f, 0.0f, 2.0f},
            {0.0f, 0.0f, 1.0f, 3.0f},
            {0.0f, 0.0f, 0.0f, 1.0f}
        }, options));

        auto gl = Eigen::toOpenGLArray(model);
        auto recovered = Eigen::fromOpenGLArray(gl);

        Eigen::Matrix<float, 4, 1> point(torch::tensor({1.0f, 2.0f, 3.0f, 1.0f}, options).unsqueeze(1));
        auto transformed = Eigen::transformOpenGLVector(model, point);
        auto perspective = Eigen::openGLPerspective(0.5f * static_cast<float>(std::acos(-1.0)), 1.0f, 1.0f, 10.0f);

        if (!matrixNear(model, recovered, 1e-5) ||
            !near(transformed(0), 2.0f, 1e-5) ||
            !near(transformed(1), 4.0f, 1e-5) ||
            !near(transformed(2), 6.0f, 1e-5) ||
            !near(perspective(0, 0), 1.0f, 1e-4) ||
            !near(perspective(1, 1), 1.0f, 1e-4)) {
            std::cerr << "OpenGLSupport mismatch" << std::endl;
            return 1;
        }
    }

    {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

        Eigen::VectorXf coeffs(torch::tensor({1.0f, -3.0f, 2.0f}, options).unsqueeze(1));
        float value = Eigen::poly_eval(coeffs, 2.0f);

        Eigen::PolynomialSolver<float> solver(coeffs);
        auto roots = solver.roots();
        at::Tensor roots_vec = roots.tensor().reshape({-1});
        at::Tensor imag_abs = torch::abs(torch::imag(roots_vec));
        at::Tensor sorted_real = std::get<0>(torch::sort(torch::real(roots_vec)));
        float root0 = sorted_real[0].template item<float>();
        float root1 = sorted_real[1].template item<float>();

        if (!near(value, 0.0f, 1e-5) ||
            imag_abs.max().template item<float>() > 1e-3f ||
            !near(root0, 1.0f, 1e-3) ||
            !near(root1, 2.0f, 1e-3)) {
            std::cerr << "Polynomials mismatch" << std::endl;
            return 1;
        }
    }

    {
        auto identity = Eigen::sparseIdentity<float>(3, 3);
        auto dense_identity = Eigen::sparseToDense(identity);
        auto random_sparse = Eigen::sparseRandom<float>(4, 4, 0.4, 7);

        Eigen::SparseMatrix<float> noisy(2, 2);
        noisy.insert(0, 0) = 1.0f;
        noisy.insert(0, 1) = 1e-3f;
        noisy.insert(1, 1) = 2.0f;
        noisy.makeCompressed();
        auto pruned = Eigen::pruneSparse(noisy, 1.0f, 1e-2f);

        if (identity.nonZeros() != 3 || random_sparse.nonZeros() <= 0 ||
            !near(Eigen::sparseTrace(identity), 3.0f, 1e-6) ||
            !near(Eigen::sparseNorm(identity), std::sqrt(3.0f), 1e-5) ||
            !near(dense_identity(2, 2), 1.0f, 1e-6) ||
            !near(pruned.coeff(0, 1), 0.0f, 1e-6)) {
            std::cerr << "SparseExtra mismatch" << std::endl;
            return 1;
        }
    }

    {
        Eigen::MatrixXf x = Eigen::MatrixXf::Zero(1, 2);
        x(0, 0) = 1.0f;
        x(0, 1) = 2.0f;

        auto erf_x = Eigen::erf(x);
        auto erfc_x = Eigen::erfc(x);
        auto lgamma_x = Eigen::lgamma(x);
        auto digamma_x = Eigen::digamma(x);
        auto trigamma_x = Eigen::polygamma(1, x);

        if (!near(Eigen::numext::erf(1.0f), std::erf(1.0f), 1e-6) ||
            !near(Eigen::numext::erfc(1.0f), std::erfc(1.0f), 1e-6) ||
            !near(Eigen::numext::lgamma(3.0f), std::lgamma(3.0f), 1e-6) ||
            !near(erf_x(0, 0), std::erf(1.0f), 1e-6) ||
            !near(erfc_x(0, 1), std::erfc(2.0f), 1e-6) ||
            !near(lgamma_x(0, 1), std::lgamma(2.0f), 1e-6) ||
            !near(digamma_x(0, 0), -0.57721566f, 2e-3) ||
            !near(digamma_x(0, 1), 0.42278434f, 2e-3) ||
            !near(trigamma_x(0, 0), static_cast<float>((std::acos(-1.0) * std::acos(-1.0)) / 6.0), 5e-3)) {
            std::cerr << "SpecialFunctions mismatch" << std::endl;
            return 1;
        }
    }

    std::cout << "Unsupported full module checks passed" << std::endl;
    return 0;
}
