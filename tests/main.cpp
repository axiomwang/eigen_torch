#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/Jacobi>
#include <Eigen/SVD>

#include <cmath>
#include <iostream>

namespace {

constexpr float kPi = 3.14159265358979323846f;

template <typename Lhs, typename Rhs>
float maxAbsDiff(const Lhs& lhs, const Rhs& rhs) {
    return (lhs - rhs).cwiseAbs().maxCoeff();
}

template <typename Lhs, typename Rhs>
bool nearMatrix(const Lhs& lhs, const Rhs& rhs, float eps = 1e-4f) {
    return maxAbsDiff(lhs, rhs) <= eps;
}

bool nearScalar(float lhs, float rhs, float eps = 1e-4f) {
    return std::abs(lhs - rhs) <= eps;
}

} // namespace

int main() {
    Eigen::DeviceManager::set_default_device(torch::kCPU);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    {
        Eigen::MatrixXf A(torch::tensor({{4.0f, 7.0f}, {2.0f, 6.0f}}, options));
        Eigen::MatrixXf B(torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}}, options));
        Eigen::MatrixXf expected_product(torch::tensor({{25.0f, 36.0f}, {20.0f, 28.0f}}, options));
        Eigen::MatrixXf expected_inverse(torch::tensor({{0.6f, -0.7f}, {-0.2f, 0.4f}}, options));

        if (!nearMatrix(A * B, expected_product) ||
            !nearScalar(A.determinant(), 10.0f) ||
            !nearMatrix(A.inverse(), expected_inverse, 1e-3f)) {
            std::cerr << "Dense matrix arithmetic mismatch" << std::endl;
            return 1;
        }
    }

    {
        Eigen::MatrixXf system(torch::tensor({{3.0f, 1.0f}, {1.0f, 2.0f}}, options));
        Eigen::VectorXf x_true(torch::tensor({2.0f, -1.0f}, options).unsqueeze(1));
        Eigen::VectorXf rhs = system * x_true;

        if (!nearMatrix(system.partialPivLu().solve(rhs), x_true) ||
            !nearMatrix(system.fullPivLu().solve(rhs), x_true) ||
            !nearMatrix(system.householderQr().solve(rhs), x_true) ||
            !nearMatrix(system.colPivHouseholderQr().solve(rhs), x_true) ||
            !nearMatrix(system.fullPivHouseholderQr().solve(rhs), x_true) ||
            !nearMatrix(system.completeOrthogonalDecomposition().solve(rhs), x_true) ||
            !nearMatrix(system.llt().solve(rhs), x_true) ||
            !nearMatrix(system.ldlt().solve(rhs), x_true)) {
            std::cerr << "Dense linear solve mismatch" << std::endl;
            return 1;
        }

        Eigen::LLT<Eigen::MatrixXf> llt(system);
        if (!nearMatrix(llt.matrixL() * llt.matrixL().transpose(), system, 1e-3f)) {
            std::cerr << "LLT reconstruction mismatch" << std::endl;
            return 1;
        }
    }

    {
        Eigen::MatrixXf diag(torch::tensor({{5.0f, 0.0f}, {0.0f, 2.0f}}, options));
        Eigen::VectorXf x_true(torch::tensor({1.5f, -2.0f}, options).unsqueeze(1));
        Eigen::VectorXf rhs = diag * x_true;

        Eigen::JacobiSVD<Eigen::MatrixXf> jacobi_svd(diag, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::BDCSVD<Eigen::MatrixXf> bdc_svd(diag, Eigen::ComputeThinU | Eigen::ComputeThinV);
        if (!nearMatrix(jacobi_svd.solve(rhs), x_true) ||
            !nearMatrix(bdc_svd.solve(rhs), x_true) ||
            !nearScalar(jacobi_svd.singularValues().sum(), 7.0f) ||
            !nearScalar(jacobi_svd.singularValues().prod(), 10.0f)) {
            std::cerr << "SVD mismatch" << std::endl;
            return 1;
        }
    }

    {
        Eigen::MatrixXf diagonal(torch::tensor({
            {1.0f, 0.0f, 0.0f},
            {0.0f, 2.0f, 0.0f},
            {0.0f, 0.0f, 4.0f}
        }, options));

        auto eig = diagonal.eigenSolver();
        auto seig = diagonal.selfAdjointEigenSolver();
        auto ceig = diagonal.complexEigenSolver();
        Eigen::MatrixXf orthogonality = seig.eigenvectors().transpose() * seig.eigenvectors();
        Eigen::MatrixXf identity = Eigen::MatrixXf::Identity(3, 3);

        if (!nearScalar(eig.eigenvalues().sum(), 7.0f) ||
            !nearScalar(eig.eigenvalues().prod(), 8.0f) ||
            !nearScalar(seig.eigenvalues().sum(), 7.0f) ||
            !nearScalar(ceig.eigenvalues().sum(), 7.0f) ||
            !nearMatrix(orthogonality, identity, 1e-4f)) {
            std::cerr << "Eigen decomposition mismatch" << std::endl;
            return 1;
        }
    }

    {
        Eigen::ArrayXXf arr(torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}}, options));
        Eigen::MatrixXf expected_square(torch::tensor({{1.0f, 4.0f}, {9.0f, 16.0f}}, options));
        Eigen::MatrixXf expected_shift(torch::tensor({{1.0f, 1.5f}, {2.0f, 2.5f}}, options));
        Eigen::MatrixXf expected_pow(torch::tensor({{1.0f, 8.0f}, {27.0f, 64.0f}}, options));

        if (!nearMatrix(arr.square().matrix(), expected_square) ||
            !nearMatrix(((arr + 1.0f) / 2.0f).matrix(), expected_shift) ||
            !nearMatrix(arr.pow(3.0f).matrix(), expected_pow)) {
            std::cerr << "Array calculation mismatch" << std::endl;
            return 1;
        }

        float raw[] = {1.0f, 2.0f, 3.0f, 4.0f};
        Eigen::Map<Eigen::Matrix<float, 2, 2>> map_mat(raw);
        map_mat(0, 1) = 5.0f;
        if (!nearScalar(map_mat.determinant(), -11.0f) || !nearScalar(raw[1], 5.0f)) {
            std::cerr << "Map mismatch" << std::endl;
            return 1;
        }

        Eigen::MatrixXf M(torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}}, options));
        M.conservativeResize(3, 3);
        Eigen::MatrixXf top_left(torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}}, options));
        if (!nearMatrix(M.topLeftCorner(2, 2), top_left) ||
            !nearScalar(M(2, 2), 0.0f)) {
            std::cerr << "Core resize/block mismatch" << std::endl;
            return 1;
        }
    }

    {
        Eigen::Vector3f unit_x = Eigen::Vector3f::UnitX();
        Eigen::Vector3f unit_y = Eigen::Vector3f::UnitY();
        Eigen::Vector3f unit_z = Eigen::Vector3f::UnitZ();
        Eigen::Vector3f cross = unit_x.cross(unit_y);
        Eigen::Vector4f homogeneous_input(torch::tensor({2.0f, 4.0f, 6.0f, 2.0f}, options).unsqueeze(1));
        Eigen::Vector3f normalized_h(torch::tensor({1.0f, 2.0f, 3.0f}, options).unsqueeze(1));
        Eigen::Vector4f expected_homogeneous(torch::tensor({1.0f, 0.0f, 0.0f, 1.0f}, options).unsqueeze(1));
        Eigen::VectorXf orthogonal = unit_x.unitOrthogonal();

        if (!nearMatrix(cross, unit_z) ||
            !nearScalar(unit_x.dot(unit_y), 0.0f) ||
            !nearMatrix(unit_x.homogeneous(), expected_homogeneous) ||
            !nearMatrix(homogeneous_input.hnormalized(), normalized_h) ||
            !nearScalar(unit_x.dot(orthogonal), 0.0f, 1e-4f) ||
            !nearScalar(orthogonal.norm(), 1.0f, 1e-4f)) {
            std::cerr << "Vector helper mismatch" << std::endl;
            return 1;
        }
    }

    {
        float s = std::sqrt(0.5f);
        Eigen::Quaternionf qz90(s, 0.0f, 0.0f, s);
        Eigen::Vector3f expected_rotated(torch::tensor({0.0f, 1.0f, 0.0f}, options).unsqueeze(1));
        Eigen::MatrixXf expected_rot(torch::tensor({
            {0.0f, -1.0f, 0.0f},
            {1.0f,  0.0f, 0.0f},
            {0.0f,  0.0f, 1.0f}
        }, options));

        if (!nearMatrix(qz90 * Eigen::Vector3f::UnitX(), expected_rotated, 1e-3f) ||
            !nearMatrix(qz90.toRotationMatrix(), expected_rot, 1e-3f) ||
            !nearScalar(qz90.norm(), 1.0f, 1e-4f)) {
            std::cerr << "Quaternion mismatch" << std::endl;
            return 1;
        }

        Eigen::AngleAxisf aa(kPi * 0.5f, Eigen::Vector3f::UnitZ());
        if (!nearMatrix(aa.toRotationMatrix(), expected_rot, 1e-3f) ||
            !nearMatrix(aa.toQuaternion() * Eigen::Vector3f::UnitX(), expected_rotated, 1e-3f)) {
            std::cerr << "AngleAxis mismatch" << std::endl;
            return 1;
        }

        Eigen::Translation3f translation(1.0f, 2.0f, 3.0f);
        Eigen::Vector3f point = Eigen::Vector3f::Ones(3, 1);
        Eigen::Vector3f translated_expected(torch::tensor({2.0f, 3.0f, 4.0f}, options).unsqueeze(1));
        if (!nearMatrix(translation * point, translated_expected) ||
            !nearMatrix(translation.inverse() * translated_expected, point)) {
            std::cerr << "Translation mismatch" << std::endl;
            return 1;
        }

        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.translate(translation);
        if (!nearMatrix(transform * point, translated_expected)) {
            std::cerr << "Transform translation mismatch" << std::endl;
            return 1;
        }

        Eigen::Affine3f rotation_transform = Eigen::Affine3f::Identity();
        rotation_transform.rotate(qz90);
        if (!nearMatrix(rotation_transform * Eigen::Vector3f::UnitX(), expected_rotated, 1e-3f)) {
            std::cerr << "Transform rotation mismatch" << std::endl;
            return 1;
        }

        Eigen::Rotation2Df rotation2d(kPi * 0.5f);
        Eigen::Matrix<float, 2, 1> expected_2d(torch::tensor({0.0f, 1.0f}, options).unsqueeze(1));
        if (!nearMatrix(rotation2d * Eigen::Vector2f::UnitX(), expected_2d, 1e-3f)) {
            std::cerr << "Rotation2D mismatch" << std::endl;
            return 1;
        }

        Eigen::Vector3f scale_factors(torch::tensor({2.0f, 3.0f, 4.0f}, options).unsqueeze(1));
        Eigen::Scaling3f scaling(scale_factors);
        Eigen::Vector3f scaled_input(torch::tensor({1.0f, 2.0f, 3.0f}, options).unsqueeze(1));
        Eigen::Vector3f scaled_expected(torch::tensor({2.0f, 6.0f, 12.0f}, options).unsqueeze(1));
        if (!nearMatrix(scaling * scaled_input, scaled_expected) ||
            !nearMatrix(scaling.toDenseMatrix(), Eigen::MatrixXf(torch::tensor({
                {2.0f, 0.0f, 0.0f},
                {0.0f, 3.0f, 0.0f},
                {0.0f, 0.0f, 4.0f}
            }, options)))) {
            std::cerr << "Scaling mismatch" << std::endl;
            return 1;
        }

        Eigen::ParametrizedLine3f line = Eigen::ParametrizedLine3f::Through(
            Eigen::Vector3f::Zero(3, 1), Eigen::Vector3f::UnitX());
        Eigen::Vector3f query(torch::tensor({2.0f, 3.0f, 0.0f}, options).unsqueeze(1));
        Eigen::Vector3f projected(torch::tensor({2.0f, 0.0f, 0.0f}, options).unsqueeze(1));
        if (!nearScalar(line.projection(query), 2.0f) ||
            !nearMatrix(line.projectionPoint(query), projected) ||
            !nearScalar(line.distance(query), 3.0f)) {
            std::cerr << "ParametrizedLine mismatch" << std::endl;
            return 1;
        }

        Eigen::Hyperplane3f plane = Eigen::Hyperplane3f::Through(
            Eigen::Vector3f::Zero(3, 1), Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitY());
        Eigen::Vector3f plane_query(torch::tensor({1.0f, 2.0f, 3.0f}, options).unsqueeze(1));
        Eigen::Vector3f plane_proj(torch::tensor({1.0f, 2.0f, 0.0f}, options).unsqueeze(1));
        if (!nearScalar(plane.signedDistance(plane_query), 3.0f, 1e-3f) ||
            !nearScalar(plane.absDistance(plane_query), 3.0f, 1e-3f) ||
            !nearMatrix(plane.projection(plane_query), plane_proj, 1e-3f)) {
            std::cerr << "Hyperplane mismatch" << std::endl;
            return 1;
        }

        Eigen::AlignedBox3f box(Eigen::Vector3f::Zero(3, 1), Eigen::Vector3f::Constant(3, 1, 2.0f));
        Eigen::AlignedBox3f box2(Eigen::Vector3f::Ones(3, 1), Eigen::Vector3f::Constant(3, 1, 3.0f));
        Eigen::Vector3f clamped_expected(torch::tensor({2.0f, 0.0f, 1.0f}, options).unsqueeze(1));
        if (!box.intersects(box2) ||
            !nearScalar(box.intersection(box2).volume(), 1.0f) ||
            !nearMatrix(box.center(), Eigen::Vector3f::Constant(3, 1, 1.0f)) ||
            !nearMatrix(box.clamp(Eigen::Vector3f(torch::tensor({4.0f, -1.0f, 1.0f}, options).unsqueeze(1))), clamped_expected)) {
            std::cerr << "AlignedBox mismatch" << std::endl;
            return 1;
        }

        Eigen::MatrixXf symmetric(torch::tensor({{4.0f, 1.0f}, {1.0f, 3.0f}}, options));
        Eigen::JacobiRotationf jacobi;
        if (!jacobi.makeJacobi(symmetric, 0, 1)) {
            std::cerr << "Jacobi rotation construction failed" << std::endl;
            return 1;
        }
        if (!nearScalar(jacobi.c() * jacobi.c() + jacobi.s() * jacobi.s(), 1.0f, 1e-4f)) {
            std::cerr << "Jacobi rotation normalization mismatch" << std::endl;
            return 1;
        }
        Eigen::MatrixXf jacobi_matrix(torch::tensor({
            {jacobi.c(), -jacobi.s()},
            {jacobi.s(),  jacobi.c()}
        }, options));
        Eigen::MatrixXf rotated = jacobi_matrix.transpose() * symmetric * jacobi_matrix;
        if (!nearScalar(rotated(0, 1), 0.0f, 1e-3f) || !nearScalar(rotated(1, 0), 0.0f, 1e-3f)) {
            std::cerr << "Jacobi rotation annihilation mismatch" << std::endl;
            return 1;
        }
    }

    std::cout << "Dense/core/geometry numerical checks passed" << std::endl;
    return 0;
}
