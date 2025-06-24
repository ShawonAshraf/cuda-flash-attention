cmake_minimum_required(VERSION 3.18)
project(cuda-flash-attention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Add include directory
target_include_directories(cuda-flash-attention PUBLIC include)

add_executable(
    cuda-flash-attention
    src/main.cu
    src/flash_attention.cu
    src/utils.cu
)

# Find CUDA and link to it
find_package(CUDA REQUIRED)
target_link_libraries(cuda-flash-attention PRIVATE CUDA::cudart) 
