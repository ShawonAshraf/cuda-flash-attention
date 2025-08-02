cmake_minimum_required(VERSION 3.18)
project(FlashAttention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")

# Main executable
add_executable(flash_attention
    src/main.cu
    src/flash_attention.cu
)

# Link CUDA libraries
target_link_libraries(flash_attention ${CUDA_LIBRARIES})

# Set properties
set_target_properties(flash_attention PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
