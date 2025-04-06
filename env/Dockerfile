# Use the official PyTorch image as base
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

ENV CC=gcc-13
ENV CXX=g++-13
ENV GCOV=gcov-13

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LIBTORCH_PATH=/usr/local/libtorch
ENV LD_LIBRARY_PATH=${LIBTORCH_PATH}/lib:${LD_LIBRARY_PATH}

# Install system dependencies and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    cuda-nvrtc-12-6 \
    cudnn-cuda-12 \
    g++-13 gcovr \
    gdb \
    git \
    lcov \
    linux-tools-generic \
    nvidia-container-toolkit \
    nvidia-cuda-toolkit \
    nvidia-utils-535 \
    sudo \
    wget \
    unzip && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 && \
    update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-13 100 && \
    rm -rf /var/lib/apt/lists/*


# Download and install libtorch (C++ binaries)
# RUN mkdir -p ${LIBTORCH_PATH} 
# COPY .cache/libtorch ${LIBTORCH_PATH}/