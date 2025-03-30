#!/usr/bin/env bash

# if .cache/libtorch exists, then we don't need to download it again
if [ ! -d ~/.cache/libtorch ]; then
    wget https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu126.zip -O .cache/libtorch.zip --progress=bar:force:noscroll
    unzip .cache/libtorch.zip -d .cache
    rm .cache/libtorch.zip
fi