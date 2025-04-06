#!/usr/bin/env bash

CONTAINER_NAME="cpp-torch-dev"
IMAGE_NAME="cpp-torch-dev"

function parse_command_line_arguments() {
    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
            -h|--help)          echo "Usage: $0 [-h|--help] [-n|--name <name>] [-i|--image <image>]";    exit 0;;
            -n | --name)        CONTAINER_NAME="$2";                                      shift ;;
            -i | --image)       IMAGE_NAME="$2";                                          shift ;;
            *)                  echo "Unknown option: $1";                                  exit 1;;
        esac
        shift
    done
}

function docker_run() {
    if [[ "$(docker ps -a | grep ${CONTAINER_NAME})" ]]; then
        docker rm ${CONTAINER_NAME}
    fi
    docker_options=(
        --env "HOME=${HOME}"
        --env "USER=${USER}"
        --gpus all
        --init
        --interactive
        --name ${CONTAINER_NAME}
        --privileged
        --tty
        --user $(id -u ${USER}):$(id -g ${USER})
        --volume "${HOME}:${HOME}"
        --volume "/etc/group:/etc/group:ro"
        --volume "/etc/passwd:/etc/passwd:ro"
        --volume "/etc/shadow:/etc/shadow:ro"
        --workdir "${HOME}"
    )
    docker run "${docker_options[@]}" ${IMAGE_NAME} /bin/bash
}

parse_command_line_arguments $@

docker_run