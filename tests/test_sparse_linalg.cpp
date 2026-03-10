#include "Eigen/Core"
#include <iostream>

int main() {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto crow_indices = torch::tensor({0, 2, 4}, torch::kInt64);
    auto col_indices = torch::tensor({0, 1, 0, 1}, torch::kInt64);
    auto values = torch::tensor({2.0, 1.0, 1.0, 2.0}, torch::kFloat32); // Positive definite
    
    // Create Sparse Matrix
    auto sparse_mat = torch::sparse_csr_tensor(crow_indices, col_indices, values, {2, 2}, options);
    auto b = torch::ones({2, 1}, options);
    
    // Test native dense solve just in case it takes CSR
    try {
        auto res = torch::linalg_solve(sparse_mat, b);
        std::cout << "torch::linalg_solve WORKS natively with CSR!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "torch::linalg_solve FAILED with CSR: " << e.what() << std::endl;
    }

    try {
        auto res = torch::linalg_cholesky(sparse_mat);
        std::cout << "torch::linalg_cholesky WORKS natively with CSR!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "torch::linalg_cholesky FAILED with CSR: " << e.what() << std::endl;
    }
    
    return 0;
}
