#!/bin/sh
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

case "$1" in 
    -d)
    HASH=$2
    FROM=$3
    TO=$4
    echo "Copy from ${HASH}:${FROM} to ${TO}"
    docker cp ${HASH}://workspaces/spatialblend-core/${FROM} ${TO}
    ;;

    -r) 
    HASH=$2
    RUN_COMMAND=${@:3}
    echo "Run from spatialblend-core workspace in docker image ${HASH} command: ${RUN_COMMAND}"
    docker exec -i ${HASH} bash -c "cd /workspaces/spatialblend-core/ && ${RUN_COMMAND}"
    ;;

    *) echo "`basename $0`: usage: [-d directory] | [-r run options (using bash from workspaces/spatialblend-core)]"
    exit 1
    ;;
esac