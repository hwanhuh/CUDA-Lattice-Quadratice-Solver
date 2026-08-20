cmake_minimum_required(VERSION 3.25...4.4)

project(lattice_qp LANGUAGES CXX CUDA)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

function(lattice_qp_python_package_root module output)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
                "from pathlib import Path; import ${module} as p; print(Path(next(iter(p.__path__))).resolve())"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE root
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Cannot locate Python CUDA package ${module}")
    endif()
    set(${output} "${root}" PARENT_SCOPE)
endfunction()

lattice_qp_python_package_root(nvidia.cuda_runtime LATTICE_QP_CUDART_ROOT)
lattice_qp_python_package_root(nvidia.cublas LATTICE_QP_CUBLAS_ROOT)
lattice_qp_python_package_root(nvidia.cusparse LATTICE_QP_CUSPARSE_ROOT)

add_library(lattice_qp_cudart SHARED IMPORTED GLOBAL)
set_target_properties(lattice_qp_cudart PROPERTIES
    IMPORTED_LOCATION "${LATTICE_QP_CUDART_ROOT}/lib/libcudart.so.12"
    INTERFACE_INCLUDE_DIRECTORIES "${LATTICE_QP_CUDART_ROOT}/include")
add_library(lattice_qp_cublas SHARED IMPORTED GLOBAL)
set_target_properties(lattice_qp_cublas PROPERTIES
    IMPORTED_LOCATION "${LATTICE_QP_CUBLAS_ROOT}/lib/libcublas.so.12"
    INTERFACE_INCLUDE_DIRECTORIES "${LATTICE_QP_CUBLAS_ROOT}/include")
add_library(lattice_qp_cusparse SHARED IMPORTED GLOBAL)
set_target_properties(lattice_qp_cusparse PROPERTIES
    IMPORTED_LOCATION "${LATTICE_QP_CUSPARSE_ROOT}/lib/libcusparse.so.12"
    INTERFACE_INCLUDE_DIRECTORIES "${LATTICE_QP_CUSPARSE_ROOT}/include")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LATTICE_QP_CUDA_ARCHITECTURES
    "70-real;75-real;80-real;86-real;89-real;90-real;90-virtual"
    CACHE STRING "CUDA architectures shipped in the lattice-qp wheel")

pybind11_add_module(_core MODULE src/lattice_qp_cuda.cu)
set_target_properties(_core PROPERTIES
    CUDA_ARCHITECTURES "${LATTICE_QP_CUDA_ARCHITECTURES}"
    INSTALL_RPATH "$ORIGIN/../nvidia/cublas/lib;$ORIGIN/../nvidia/cuda_runtime/lib;$ORIGIN/../nvidia/cusparse/lib"
    INSTALL_RPATH_USE_LINK_PATH FALSE
)
target_compile_definitions(_core PRIVATE NDEBUG=1)
target_compile_options(_core PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-O3;-g0;-fvisibility=hidden>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;--extended-lambda;--expt-relaxed-constexpr;-Wno-deprecated-gpu-targets;-Xcompiler=-fPIC,-fvisibility=hidden>"
)
target_link_libraries(_core PRIVATE
    lattice_qp_cudart
    lattice_qp_cublas
    lattice_qp_cusparse
)

install(TARGETS _core LIBRARY DESTINATION lattice_qp)
