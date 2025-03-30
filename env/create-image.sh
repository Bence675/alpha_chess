#!/usr/bin/env bash

ENV_PATH="."

function parse_command_line_arguments() {
    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
            -h|--help)          echo "Usage: $0 [-h|--help] [-p|--path <path>]";    exit 0;;
            -p | --path)        ENV_PATH="$2";                                      shift ;;
            *)                  echo "Unknown option: $1";                          exit 1;;
        esac
        shift
    done
}

parse_command_line_arguments $@

# ./env/download_libtorch.sh

docker build -t cpp-torch-dev ${ENV_PATH}