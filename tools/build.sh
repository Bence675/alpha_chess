#! /usr/bin/bash

cd ../build
mkdir debug
mkdir release
mkdir perf_build

repo_path=$(pwd | sed 's/\/build//g')
# Run cmake to generate the build files
cmake -S $repo_path -B $repo_path/build/debug  -DCMAKE_PREFIX_PATH=$repo_path/include/libtorch  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer  -fprofile-arcs -ftest-coverage" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build $repo_path/build/debug  -j 24

cmake -S $repo_path -B $repo_path/build/release -DCMAKE_PREFIX_PATH=$repo_path/include/libtorch  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O2" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build $repo_path/build/release  -j 24

cmake -S $repo_path -B $repo_path/build/perf_build -DCMAKE_PREFIX_PATH=$repo_path/include/libtorch  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-O0" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build $repo_path/build/perf_build  -j 24

# Build the project
# make -j 24