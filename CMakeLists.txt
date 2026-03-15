cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(Eigen VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Option to use Python to find LibTorch if Torch_DIR is not set natively
option(USE_PYTHON_TORCH "Use Python's torch package to find LibTorch" ON)

if(USE_PYTHON_TORCH)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(TORCH_CMAKE_PREFIX_PATH)
        list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
        message(STATUS "Appended Python Torch cmake_prefix_path: ${TORCH_CMAKE_PREFIX_PATH}")
    endif()
endif()

find_package(Torch REQUIRED)

# Eigen is a header-only or template-heavy static library conceptually, 
# but we define it as an INTERFACE library since it just wraps Torch calls in headers
add_library(Eigen INTERFACE)
target_include_directories(Eigen INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(Eigen INTERFACE ${TORCH_LIBRARIES})
set_property(TARGET Eigen PROPERTY CXX_STANDARD 17)

# Testing
enable_testing()
add_executable(test_eigentorch tests/main.cpp)
target_link_libraries(test_eigentorch PRIVATE Eigen)
add_test(NAME core_test COMMAND test_eigentorch)

# --- Install & Export Rules ---

include(CMakePackageConfigHelpers)

# Install headers
install(DIRECTORY include/Eigen DESTINATION include)

# Install targets
install(TARGETS Eigen
    EXPORT EigenTargets
    INCLUDES DESTINATION include
)

# Export the targets file for tree imports
install(EXPORT EigenTargets
    FILE EigenTargets.cmake
    NAMESPACE Eigen::
    DESTINATION lib/cmake/Eigen
)

# Generate and install the Config and Version files
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EigenConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/EigenConfig.cmake"
    INSTALL_DESTINATION lib/cmake/Eigen
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/EigenConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/EigenConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/EigenConfigVersion.cmake"
    DESTINATION lib/cmake/Eigen
)

add_executable(test_sparse_linalg tests/test_sparse_linalg.cpp)
target_link_libraries(test_sparse_linalg PRIVATE Eigen)
add_test(NAME sparse_linalg_test COMMAND test_sparse_linalg)

add_executable(test_api_compat tests/test_api_compat.cpp)
target_link_libraries(test_api_compat PRIVATE Eigen)
add_test(NAME api_compat_test COMMAND test_api_compat)
