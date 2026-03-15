#include <unsupported/Eigen/AutoDiff>
#include <unsupported/Eigen/BVH>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/EulerAngles>
#include <unsupported/Eigen/FFT>
#include <unsupported/Eigen/NNLS>
#include <unsupported/Eigen/Splines>

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;

bool near(float lhs, float rhs, float eps = 1e-3f) {
    return std::abs(lhs - rhs) <= eps;
}

bool complexNear(const std::complex<float>& lhs, const std::complex<float>& rhs, float eps = 1e-3f) {
    return std::abs(lhs - rhs) <= eps;
}

bool matrixNear(const Eigen::MatrixXf& lhs, const Eigen::MatrixXf& rhs, float eps = 1e-3f) {
    return (lhs - rhs).cwiseAbs().maxCoeff() <= eps;
}

} // namespace

int main() {
    Eigen::DeviceManager::set_default_device(torch::kCPU);

    {
        Eigen::AutoDiffScalar<Eigen::MatrixXf> x(0.5f);
        auto y = Eigen::sin(x) + Eigen::exp(x);
        y.backward();
        float value = y.value();
        float grad = x.grad().template item<float>();
        float expected_value = std::sin(0.5f) + std::exp(0.5f);
        float expected = std::cos(0.5f) + std::exp(0.5f);
        if (!near(value, expected_value, 1e-3f) || !near(grad, expected, 1e-3f)) {
            std::cerr << "AutoDiff gradient mismatch" << std::endl;
            return 1;
        }

        Eigen::AutoDiffScalar<Eigen::MatrixXf> z(2.0f);
        auto polynomial = z * z + 3.0f * z - 1.0f;
        polynomial.backward();
        if (!near(polynomial.value(), 9.0f, 1e-3f) || !near(z.grad().template item<float>(), 7.0f, 1e-3f)) {
            std::cerr << "AutoDiff polynomial mismatch" << std::endl;
            return 1;
        }

        if (!near(x.derivatives().sum(), grad, 1e-3f)) {
            std::cerr << "AutoDiff derivatives mismatch" << std::endl;
            return 1;
        }
    }

    {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        Eigen::EulerAnglesf euler(kPi * 0.5f, 0.0f, 0.0f);
        Eigen::Matrix<float, 3, 1> rotated = euler * Eigen::Vector3f::UnitX();
        Eigen::MatrixXf expected_rot(torch::tensor({
            {0.0f, -1.0f, 0.0f},
            {1.0f,  0.0f, 0.0f},
            {0.0f,  0.0f, 1.0f}
        }, options));
        Eigen::Matrix<float, 3, 1> expected_vec(torch::tensor({0.0f, 1.0f, 0.0f}, options).unsqueeze(1));
        Eigen::EulerAnglesf recovered(euler.toRotationMatrix());
        Eigen::EulerAnglesf from_q(euler.toQuaternion());
        if (!matrixNear(euler.toRotationMatrix(), expected_rot, 1e-3f) ||
            !matrixNear(rotated, expected_vec, 1e-3f) ||
            !near(recovered.alpha(), kPi * 0.5f, 1e-3f) ||
            !near(from_q.alpha(), kPi * 0.5f, 1e-3f)) {
            std::cerr << "EulerAngles conversion mismatch" << std::endl;
            return 1;
        }

        Eigen::Matrix<float, 3, 1> inverse_rotated = euler.inverse() * expected_vec;
        if (!matrixNear(inverse_rotated, Eigen::Vector3f::UnitX(), 1e-3f)) {
            std::cerr << "EulerAngles inverse mismatch" << std::endl;
            return 1;
        }
    }

    {
        Eigen::FFTf fft;
        fft.SetFlag(Eigen::FFTf::HalfSpectrum);
        std::vector<float> signal = {1.0f, 1.0f, 1.0f, 1.0f};
        std::vector<std::complex<float>> spectrum;
        std::vector<float> recovered;
        fft.fwd(spectrum, signal);
        fft.inv(recovered, spectrum, static_cast<int>(signal.size()));
        if (spectrum.size() != 3 || recovered.size() != signal.size()) {
            std::cerr << "FFT shape mismatch" << std::endl;
            return 1;
        }
        if (!complexNear(spectrum[0], std::complex<float>(4.0f, 0.0f)) ||
            !complexNear(spectrum[1], std::complex<float>(0.0f, 0.0f)) ||
            !complexNear(spectrum[2], std::complex<float>(0.0f, 0.0f))) {
            std::cerr << "FFT frequency mismatch" << std::endl;
            return 1;
        }
        for (size_t index = 0; index < signal.size(); ++index) {
            if (!near(recovered[index], signal[index], 1e-3f)) {
                std::cerr << "FFT reconstruction mismatch" << std::endl;
                return 1;
            }
        }

        fft.ClearFlag(Eigen::FFTf::HalfSpectrum);
        std::vector<std::complex<float>> complex_signal = {
            {1.0f, 0.0f}, {0.0f, 1.0f}, {-1.0f, 0.0f}, {0.0f, -1.0f}
        };
        std::vector<std::complex<float>> full_spectrum;
        std::vector<std::complex<float>> recovered_complex;
        fft.fwd(full_spectrum, complex_signal);
        fft.inv(recovered_complex, full_spectrum, static_cast<int>(complex_signal.size()));
        for (size_t index = 0; index < complex_signal.size(); ++index) {
            if (!complexNear(recovered_complex[index], complex_signal[index], 1e-3f)) {
                std::cerr << "Complex FFT reconstruction mismatch" << std::endl;
                return 1;
            }
        }
    }

    {
        Eigen::Tensor<float, 3> tensor({2, 3, 4});
        tensor.setConstant(2.0f);
        tensor(1, 2, 3) = 5.0f;
        auto reshaped = tensor.reshape<2>({4, 6});
        auto shuffled = tensor.shuffle({1, 0, 2});
        Eigen::Tensor<float, 2> lhs({2, 2});
        lhs(0, 0) = 1.0f;
        lhs(0, 1) = 2.0f;
        lhs(1, 0) = 3.0f;
        lhs(1, 1) = 4.0f;
        Eigen::Tensor<float, 2> rhs({2, 2});
        rhs.setZero();
        rhs(0, 0) = 1.0f;
        rhs(1, 1) = 1.0f;
        auto contracted = lhs.contract(rhs, {{1, 0}});
        if (tensor.dimension(0) != 2 || tensor.dimension(1) != 3 || tensor.dimension(2) != 4 ||
            !near(tensor(1, 2, 3), 5.0f, 1e-6f) || reshaped.dimension(0) != 4 ||
            reshaped.dimension(1) != 6 || shuffled.dimension(0) != 3 ||
            !near(tensor.sum(), 51.0f, 1e-5f) || !near(tensor.mean(), 51.0f / 24.0f, 1e-5f) ||
            !near(lhs.square().sum(), 30.0f, 1e-5f) || !near(contracted(1, 0), 3.0f, 1e-5f)) {
            std::cerr << "Tensor API mismatch" << std::endl;
            return 1;
        }
    }

    {
        Eigen::AlignedBox3f box1(Eigen::Vector3f::Zero(3, 1), Eigen::Vector3f::Ones(3, 1));
        Eigen::AlignedBox3f box2(Eigen::Vector3f::Constant(3, 1, 0.5f), Eigen::Vector3f::Constant(3, 1, 1.5f));
        Eigen::AlignedBox3f box3(Eigen::Vector3f::Constant(3, 1, 3.0f), Eigen::Vector3f::Constant(3, 1, 4.0f));
        std::vector<Eigen::AlignedBox3f> boxes = {box1, box2, box3};

        Eigen::KdBVH<float, 3, Eigen::AlignedBox3f> bvh;
        bvh.init(boxes.begin(), boxes.end());
        auto hits = bvh.intersectingObjects(Eigen::AlignedBox3f(
            Eigen::Vector3f::Constant(3, 1, 0.75f),
            Eigen::Vector3f::Constant(3, 1, 0.9f)));
        auto point_hits = bvh.intersectingObjects(Eigen::Vector3f::Constant(3, 1, 3.5f));
        if (!bvh.getRoot() || !bvh.intersects(box1) || hits.size() != 2 || point_hits.size() != 1 ||
            !bvh.getRoot()->bounds.contains(Eigen::Vector3f::Constant(3, 1, 4.0f))) {
            std::cerr << "BVH query mismatch" << std::endl;
            return 1;
        }
    }

    {
        std::vector<Eigen::Vector3f> ctrls = {
            Eigen::Vector3f::Zero(3, 1),
            Eigen::Vector3f::Ones(3, 1)
        };
        Eigen::Spline3f spline({}, ctrls);
        Eigen::Vector3f quarter = spline(0.25f);
        Eigen::Vector3f three_quarter = spline(0.75f);
        auto derivs = spline.derivatives(0.5f, 1);
        auto basis = spline.basisFunctions(0.25f);
        if (derivs.size() != 2 || !near(quarter.x(), 0.25f, 1e-3f) || !near(three_quarter.x(), 0.75f, 1e-3f) ||
            !near(derivs[1].x(), 1.0f, 5e-2f) || !near(basis.sum(), 1.0f, 1e-4f) || spline.span(0.5f) != 1) {
            std::cerr << "Spline evaluation mismatch" << std::endl;
            return 1;
        }
    }

    {
        Eigen::MatrixXf A = Eigen::MatrixXf::Identity(2, 2);
        Eigen::MatrixXf b(2, 1);
        b(0, 0) = -1.0f;
        b(1, 0) = 2.0f;
        Eigen::NNLS<Eigen::MatrixXf> nnls;
        nnls.setTolerance(1e-6f).setMaxIterations(128).compute(A);
        Eigen::MatrixXf x = nnls.solve(b);
        if (nnls.info() != Eigen::Success || x(0, 0) < -1e-6f || !near(x(0, 0), 0.0f, 1e-3f) ||
            !near(x(1, 0), 2.0f, 1e-3f) || nnls.iterations() <= 0 || !near(nnls.residualNorm(), 1.0f, 1e-3f)) {
            std::cerr << "NNLS solve mismatch" << std::endl;
            return 1;
        }

        Eigen::MatrixXf A_over(torch::tensor({
            {1.0f, 0.0f},
            {0.0f, 1.0f},
            {1.0f, 1.0f}
        }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)));
        Eigen::MatrixXf b_over(torch::tensor({1.0f, 1.0f, 2.0f}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).unsqueeze(1));
        nnls.setTolerance(1e-6f).setMaxIterations(256).compute(A_over);
        Eigen::MatrixXf x_over = nnls.solve(b_over);
        if (nnls.info() != Eigen::Success || !near(x_over(0, 0), 1.0f, 1e-3f) || !near(x_over(1, 0), 1.0f, 1e-3f) ||
            nnls.residualNorm() > 1e-3f) {
            std::cerr << "NNLS overdetermined mismatch" << std::endl;
            return 1;
        }
    }

    std::cout << "Unsupported module compatibility checks passed" << std::endl;
    return 0;
}