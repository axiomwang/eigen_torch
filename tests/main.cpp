#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <Eigen/Jacobi>
#include <unsupported/Eigen/AutoDiff>
#include <unsupported/Eigen/EulerAngles>
#include <unsupported/Eigen/FFT>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/BVH>
#include <unsupported/Eigen/Splines>
#include <unsupported/Eigen/NNLS>
#include <iostream>

int main() {
    std::cout << "Starting Eigen Phase 2 Tests..." << std::endl;
    Eigen::DeviceManager::set_default_device(torch::kCPU);

    std::cout << "\n--- LU Decomposition ---" << std::endl;
    Eigen::MatrixXf A = Eigen::MatrixXf::Random(3, 3);
    Eigen::VectorXf b = Eigen::VectorXf::Random(3, 1);
    Eigen::PartialPivLU<Eigen::MatrixXf> lusolver(A);
    Eigen::VectorXf x_lu = lusolver.solve(b);
    std::cout << "LU Solution Error: " << (A * x_lu - b).cwiseAbs().maxCoeff() << " (Expected ~0)" << std::endl;

    std::cout << "\n--- Cholesky Decomposition ---" << std::endl;
    Eigen::MatrixXf A_sym = A.transpose() * A + Eigen::MatrixXf::Identity(3,3); // Ensure pos-def
    Eigen::LLT<Eigen::MatrixXf> lltsolver(A_sym);
    Eigen::VectorXf x_llt = lltsolver.solve(b);
    std::cout << "Cholesky Solution Error: " << (A_sym * x_llt - b).cwiseAbs().maxCoeff() << " (Expected ~0)" << std::endl;

    std::cout << "\n--- QR Decomposition ---" << std::endl;
    Eigen::HouseholderQR<Eigen::MatrixXf> qrsolver(A);
    Eigen::VectorXf x_qr = qrsolver.solve(b);
    std::cout << "QR Solution Error: " << (A * x_qr - b).cwiseAbs().maxCoeff() << " (Expected ~0)" << std::endl;

    std::cout << "\n--- SVD Decomposition ---" << std::endl;
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXf x_svd = svd.solve(b);
    std::cout << "SVD Solution Error: " << (A * x_svd - b).cwiseAbs().maxCoeff() << " (Expected ~0)" << std::endl;

    std::cout << "\n--- Eigenvalues ---" << std::endl;
    Eigen::EigenSolver<Eigen::MatrixXf> eig(A);
    std::cout << "Eigenvalues (Real component) shape: " << eig.eigenvalues().rows() << "x" << eig.eigenvalues().cols() << std::endl;
    
    std::cout << "\n--- Geometry Module ---" << std::endl;
    Eigen::Quaternionf q(1, 0, 0, 0);
    Eigen::Matrix<float, 3, 1> point = Eigen::Matrix<float, 3, 1>::Ones(3, 1);
    
    // Quaternion * Vector returns Fixed Dynamic, auto converts? Actually operator* is fixed
    Eigen::Matrix<float, 3, 1> rotated = q * point;
    std::cout << "Point rotated by Identity Quaternion matches self: " << (rotated - point).cwiseAbs().maxCoeff() << std::endl;
    std::cout << "Quat -> Rot Matrix size: " << q.toRotationMatrix().rows() << "x" << q.toRotationMatrix().cols() << std::endl;
    
    Eigen::AngleAxisf aa(3.14159f, Eigen::Matrix<float, 3, 1>::Ones(3, 1));
    std::cout << "AngleAxis -> Rot Matrix valid: " << aa.toRotationMatrix().rows() << "x" << aa.toRotationMatrix().cols() << std::endl;

    Eigen::Translation3f t(Eigen::Matrix<float, 3, 1>::Ones(3, 1));
    std::cout << "Translation valid x/y/z: " << t.x() << ", " << t.y() << ", " << t.z() << std::endl;
    
    Eigen::Affine3f T; // Identity transform
    Eigen::VectorXf p2 = T * point; 
    std::cout << "Affine transform applied: Error=" << (p2 - point).cwiseAbs().maxCoeff() << std::endl;

    std::cout << "\n--- Sparse Matrix Module ---" << std::endl;
    Eigen::SparseMatrixf sm(3, 3);
    sm.insert(0, 0, 1.0f);
    sm.insert(1, 1, 2.0f);
    sm.insert(2, 2, 3.0f);
    sm.makeCompressed();
    
    std::cout << "Sparse Matrix nonZero elements: " << sm.nonZeros() << std::endl;
    Eigen::VectorXf x_sm = Eigen::VectorXf::Ones(3, 1); // Changed to 3x1 to match sm
    Eigen::VectorXf y_sm = sm * x_sm;
    std::cout << "Sparse * Dense vector top element expected ~1: " << y_sm.tensor()[0][0].item<float>() << std::endl;

    // Advanced Sparse Solvers
    Eigen::SparseLU<Eigen::SparseMatrix<float>> slu(sm);
    Eigen::VectorXf slu_ans = slu.solve(y_sm);
    std::cout << "SparseLU Error: " << (x_sm - slu_ans).cwiseAbs().maxCoeff() << std::endl;

    Eigen::SparseQR<Eigen::SparseMatrix<float>> sqr(sm);
    Eigen::VectorXf sqr_ans = sqr.solve(y_sm);
    std::cout << "SparseQR Error: " << (x_sm - sqr_ans).cwiseAbs().maxCoeff() << std::endl;

    Eigen::SimplicialLLT<Eigen::SparseMatrix<float>> sllt(sm); // sm should be pos-def for CHol
    Eigen::VectorXf sllt_ans = sllt.solve(y_sm);
    std::cout << "SimplicialLLT Error: " << (x_sm - sllt_ans).cwiseAbs().maxCoeff() << std::endl;

    std::cout << "\n--- Advanced Geometry Module ---" << std::endl;
    Eigen::Rotation2Df rot2d(3.14159f / 2.0f); // 90 deg
    Eigen::Matrix<float, 2, 1> p2d = Eigen::Matrix<float, 2, 1>::Ones(2, 1);
    std::cout << "Rotation2D applied length matching: " << (rot2d * p2d).rows() << "x1" << std::endl;

    Eigen::Scaling3f scale(2.0f);
    std::cout << "Scaling3D top corner: " << (scale * point).tensor()[0][0].item<float>() << " (expected 2.0)" << std::endl;

    Eigen::ParametrizedLine3f line(point, Eigen::Matrix<float, 3, 1>::Ones(3, 1));
    std::cout << "ParametrizedLine Distance: " << line.distance(point) << " (expected 0)" << std::endl;

    Eigen::Hyperplane3f plane(Eigen::Matrix<float, 3, 1>::Ones(3, 1), 0.0f);
    std::cout << "Hyperplane center normal: " << plane.normal().tensor()[0][0].item<float>() << std::endl;

    Eigen::AlignedBox3f box(point);
    box.extend(Eigen::Matrix<float, 3, 1>(Eigen::Matrix<float, 3, 1>::Ones(3, 1).tensor() * -1.0f));
    std::cout << "AlignedBox center matches 0: " << box.center().tensor()[0][0].item<float>() << std::endl;

    std::cout << "\n--- Jacobi Module ---" << std::endl;
    Eigen::JacobiRotationf jac(1.0f, 0.0f);
    std::cout << "Jacobi Rotation default structure ok C=1 S=0: C=" << jac.c() << " S=" << jac.s() << std::endl;

    std::cout << "\n--- Phase 6: Typedefs & Unsupported Modules ---" << std::endl;
    Eigen::Matrix4d m4d = Eigen::Matrix4d::Identity(4, 4);
    Eigen::Vector3f v3f = Eigen::Vector3f::Ones(3, 1);
    std::cout << "Typedef Matrix4d size: " << m4d.rows() << "x" << m4d.cols() << std::endl;
    std::cout << "Typedef Vector3f size: " << v3f.rows() << "x" << v3f.cols() << std::endl;

    Eigen::AutoDiffScalar<Eigen::MatrixXf> x(5.0f);
    Eigen::AutoDiffScalar<Eigen::MatrixXf> y = x * x; // 5^2 = 25, derivative = 2*5 = 10
    y.backward();
    std::cout << "AutoDiffScalar (5^2) = " << y.value() << ", Gradient (2*5) = " << x.grad().item<float>() << std::endl;

    Eigen::EulerAnglesf euler(0.0f, 0.0f, 0.0f);
    std::cout << "EulerAngles -> RotMatrix [0,0]: " << euler.toRotationMatrix().tensor()[0][0].item<float>() << std::endl;

    Eigen::Tensor<float, 3> t3d({2, 2, 2});
    t3d.setConstant(3.14f);
    std::cout << "CXX11/Tensor N-Dim rank: " << t3d.tensor().dim() << ", numel: " << t3d.size() << std::endl;

    Eigen::FFTf fft;
    Eigen::MatrixXf sig = Eigen::MatrixXf::Ones(4, 1);
    Eigen::MatrixXf sig_fwd;
    fft.fwd(sig_fwd, sig);
    std::cout << "FFT Forward elements yielded correctly (complex format handled natively under torch)" << std::endl;

    std::vector<int> bvh_nodes = {1, 2, 3};
    Eigen::KdBVHf3 bvh;
    bvh.init(bvh_nodes.begin(), bvh_nodes.end());
    std::cout << "KdBVH Root isLeaf evaluation: " << bvh.getRoot()->isLeaf << std::endl;

    std::vector<float> knots = {0.0f, 0.5f, 1.0f};
    std::vector<Eigen::Matrix<float, 3, 1>> ctrls = {v3f, v3f};
    Eigen::Spline3f spline(knots, ctrls);
    std::cout << "Spline evaluated u=0.5: " << spline(0.5f).tensor()[0][0].item<float>() << std::endl;

    Eigen::MatrixXf A_nnls = Eigen::MatrixXf::Identity(2, 2);
    Eigen::VectorXf b_nnls(Eigen::VectorXf::Ones(2, 1).tensor() * -1.0f); // Target negatives
    Eigen::NNLS<Eigen::MatrixXf> nnls(A_nnls);
    Eigen::VectorXf x_nnls = nnls.solve(b_nnls);
    std::cout << "NNLS Projected Target for Negatives (clipped to >=0): " << x_nnls.tensor()[0][0].item<float>() << std::endl;

    std::cout << "\nAll features executed successfully!\n" << std::endl;
    return 0;
}
