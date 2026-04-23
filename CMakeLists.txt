cmake_minimum_required(VERSION 3.18)

project(MyLLMEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(pybind11_DIR "~/test1/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "80;86")


set(SOURCES
    # src/main.cpp
    src/model.cu
    src/kernels.cu
    src/bindings.cpp
)

pybind11_add_module(llm_engine ${SOURCES})
