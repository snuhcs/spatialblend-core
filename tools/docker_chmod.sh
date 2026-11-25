#!/bin/bash
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

# Uses Docker to chmod files/directories as root.
# Useful when you don't have sudo on the host.
#
# Usage:
#   ./docker_chmod.sh <mode> <path> [path...]
#   ./docker_chmod.sh -R <mode> <path> [path...]
#
# Examples:
#   ./docker_chmod.sh 755 ./tools/build_c_api.py
#   ./docker_chmod.sh -R 777 ./output
#   ./docker_chmod.sh +x ./tools/*.sh

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 [-R] <mode> <path> [path...]"
  echo "Example: $0 755 ./some/file"
  echo "Example: $0 -R 755 ./some/dir"
  exit 1
fi

# Parse optional -R flag
RECURSIVE=""
if [ "$1" = "-R" ] || [ "$1" = "-r" ]; then
  RECURSIVE="-R"
  shift
fi

MODE="$1"
shift

if [ $# -eq 0 ]; then
  echo "Error: no paths specified"
  exit 1
fi

for target in "$@"; do
  target="$(realpath "$target")"

  if [ -d "$target" ]; then
    docker run --rm -v "${target}:/mnt/target" alpine:latest \
      chmod ${RECURSIVE} "${MODE}" /mnt/target
  elif [ -f "$target" ]; then
    parent_dir="$(dirname "$target")"
    filename="$(basename "$target")"
    docker run --rm -v "${parent_dir}:/mnt/target" alpine:latest \
      chmod ${RECURSIVE} "${MODE}" "/mnt/target/${filename}"
  else
    echo "Warning: '${target}' does not exist, skipping"
    continue
  fi

  echo "chmod ${RECURSIVE} ${MODE} ${target}"
  ls -ld "$target"
done
