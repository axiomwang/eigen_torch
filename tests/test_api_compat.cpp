#include <Eigen/Eigen>
#include <iostream>

int main() {
    Eigen::DeviceManager::set_default_device(torch::kCPU);

    Eigen::MatrixXf A = Eigen::MatrixXf::Random(4, 4);
    Eigen::VectorXf b = Eigen::VectorXf::Ones(4, 1);

    Eigen::VectorXf x_lu = A.partialPivLu().solve(b);
    Eigen::VectorXf x_qr = A.householderQr().solve(b);
    Eigen::VectorXf x_cod = A.completeOrthogonalDecomposition().solve(b);

    Eigen::MatrixXf spd = A.transpose() * A + Eigen::MatrixXf::Identity(4, 4);
    Eigen::VectorXf x_llt = spd.llt().solve(b);
    Eigen::VectorXf x_ldlt = spd.ldlt().solve(b);

    Eigen::VectorXf x_svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

    auto eig = A.eigenSolver();
    auto seig = spd.selfAdjointEigenSolver();
    auto ceig = A.complexEigenSolver();

    Eigen::MatrixXf M(3, 3);
    M.setIdentity();
    M.conservativeResize(4, 4);
    M.topLeftCorner(2, 2).setOnes();

    Eigen::ArrayXXf arr = Eigen::ArrayXXf::Ones(2, 2);
    Eigen::ArrayXXf arr_prod = arr.array().cwiseProduct(arr.array());
    Eigen::ArrayXXf arr_elem = arr.array() * arr.array();
    Eigen::ArrayXXf arr_shift = (arr.array() + 2.0f) / 2.0f;
    Eigen::MatrixXf arr_back = arr_elem.matrix();

    float raw[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Eigen::Map<Eigen::Matrix<float, 2, 2>> map_mat(raw);
    float det = map_mat.determinant();

    Eigen::Vector3f v1 = Eigen::Vector3f::Ones(3, 1);
    Eigen::Vector3f v2 = Eigen::Vector3f::UnitY();
    Eigen::Vector3f v_cross = v1.cross(v2);
    Eigen::Vector4f hv = Eigen::Vector4f::Ones(4, 1);
    Eigen::Vector3f hv_n = hv.hnormalized();
    Eigen::Vector4f v_h = v1.homogeneous();
    Eigen::Vector3f ortho = v1.unitOrthogonal();

    Eigen::Quaternionf q = Eigen::Quaternionf::Identity();
    Eigen::Quaternionf q2(0.9238795f, 0.0f, 0.3826834f, 0.0f);
    Eigen::Quaternionf q_mid = q.slerp(0.5f, q2);

    Eigen::AngleAxisf aa(q2);
    Eigen::AngleAxisf aa_inv = aa.inverse();

    Eigen::Affine3f T = Eigen::Affine3f::Identity();
    Eigen::Translation3f t(1.0f, 2.0f, 3.0f);
    T.translate(t);
    T.pretranslate(Eigen::Vector3f::Ones(3, 1));
    T.rotate(q2);
    T.prerotate(aa_inv);
    T.scale(2.0f);
    T.prescale(Eigen::Vector3f::Constant(3, 1, 0.5f));
    Eigen::Vector3f moved = T * Eigen::Vector3f::Ones(3, 1);

    Eigen::ParametrizedLine3f line = Eigen::ParametrizedLine3f::Through(
        Eigen::Vector3f::Zero(3, 1), Eigen::Vector3f::Ones(3, 1));
    line.normalize();

    Eigen::Hyperplane3f plane = Eigen::Hyperplane3f::Through(
        Eigen::Vector3f::Zero(3, 1),
        Eigen::Vector3f::UnitX(),
        Eigen::Vector3f::UnitY());
    plane.normalize();

    Eigen::AlignedBox3f box(Eigen::Vector3f::Zero(3, 1), Eigen::Vector3f::Ones(3, 1));
    Eigen::AlignedBox3f box2(Eigen::Vector3f::Constant(3, 1, 0.5f), Eigen::Vector3f::Constant(3, 1, 2.0f));
    bool intersects = box.intersects(box2);
    Eigen::AlignedBox3f box_inter = box.intersection(box2);
    Eigen::Vector3f clamped = box.clamp(Eigen::Vector3f::Constant(3, 1, 2.5f));

    Eigen::SparseMatrixf sparseA(3, 3);
    sparseA.insert(0, 0) = 4.0f;
    sparseA.insert(0, 1) = 1.0f;
    sparseA.insert(1, 0) = 1.0f;
    sparseA.insert(1, 1) = 4.0f;
    sparseA.insert(1, 2) = 1.0f;
    sparseA.insert(2, 1) = 1.0f;
    sparseA.insert(2, 2) = 3.0f;
    sparseA.makeCompressed();
    sparseA.prune();

    Eigen::VectorXf sb = Eigen::VectorXf::Ones(3, 1);
    Eigen::VectorXf srhs = sparseA * sb;
    Eigen::ComputationInfo sparse_info = Eigen::InvalidInput;

    Eigen::SparseLU<Eigen::SparseMatrixf> sparse_lu;
    sparse_lu.analyzePattern(sparseA);
    sparse_lu.factorize(sparseA);
    sparse_info = sparse_lu.info();
    Eigen::VectorXf sx_lu = sparse_lu.solve(srhs);
    auto luL = sparse_lu.matrixL();
    auto luU = sparse_lu.matrixU();

    Eigen::SparseQR<Eigen::SparseMatrixf> sparse_qr;
    sparse_qr.compute(sparseA);
    sparse_info = sparse_qr.info();
    Eigen::VectorXf sx_qr = sparse_qr.solve(srhs);
    auto qrQ = sparse_qr.matrixQ();
    auto qrR = sparse_qr.matrixR();

    Eigen::SimplicialLLT<Eigen::SparseMatrixf> sparse_llt;
    sparse_llt.compute(sparseA);
    sparse_info = sparse_llt.info();
    Eigen::VectorXf sx_llt = sparse_llt.solve(srhs);
    auto lltL = sparse_llt.matrixL();

    Eigen::SimplicialLDLT<Eigen::SparseMatrixf> sparse_ldlt;
    sparse_ldlt.compute(sparseA);
    sparse_info = sparse_ldlt.info();
    Eigen::VectorXf sx_ldlt = sparse_ldlt.solve(srhs);
    auto ldltD = sparse_ldlt.vectorD();

    Eigen::SparseMatrix<float, Eigen::RowMajor> sparseRow(sparseA.tensor());
    Eigen::VectorXf sx_row = sparseRow * sb;

    std::cout << "compat rows: "
              << x_lu.rows() << ", "
              << x_qr.rows() << ", "
              << x_cod.rows() << ", "
              << x_llt.rows() << ", "
              << x_ldlt.rows() << ", "
              << x_svd.rows() << "\n";
    std::cout << "eig dims: "
              << eig.eigenvalues().rows() << ", "
              << seig.eigenvalues().rows() << ", "
              << ceig.eigenvalues().rows() << "\n";
    std::cout << "array prod sum: " << arr_prod.sum() << "\n";
    std::cout << "array elem sum: " << arr_elem.sum() << "\n";
    std::cout << "array shift sum: " << arr_shift.sum() << "\n";
    std::cout << "array back sum: " << arr_back.sum() << "\n";
    std::cout << "map determinant: " << det << "\n";
    std::cout << "geom norms: "
              << v_cross.norm() << ", "
              << hv_n.norm() << ", "
              << v_h.norm() << ", "
              << ortho.norm() << ", "
              << q_mid.norm() << ", "
              << moved.norm() << ", "
              << line.distance(Eigen::Vector3f::Ones(3, 1)) << ", "
              << plane.absDistance(Eigen::Vector3f::Zero(3, 1)) << ", "
              << box_inter.volume() << ", "
              << clamped.norm() << "\n";
    std::cout << "sparse rows/cols/info: "
              << sparse_lu.rows() << "x" << sparse_lu.cols() << ", "
              << static_cast<int>(sparse_info) << "\n";
    std::cout << "sparse norms: "
              << sx_lu.norm() << ", "
              << sx_qr.norm() << ", "
              << sx_llt.norm() << ", "
              << sx_ldlt.norm() << ", "
              << sx_row.norm() << ", "
              << luL.norm() + luU.norm() + qrQ.norm() + qrR.norm() + lltL.norm() + ldltD.norm()
              << "\n";
    std::cout << "boxes intersect: " << intersects << "\n";

    if (x_lu.rows() != 4 || x_qr.rows() != 4 || x_cod.rows() != 4 ||
        x_llt.rows() != 4 || x_ldlt.rows() != 4 || x_svd.rows() != 4 ||
        !intersects || arr_elem.sum() != arr_prod.sum() || arr_back.sum() != arr_elem.sum() ||
        sparse_lu.rows() != 3 || sparse_lu.cols() != 3) {
        return 1;
    }

    return 0;
}
