#!/bin/bash
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

# ADB Utility Library - shared helpers for ADB forwarding scripts

# Configuration
ADB="${ADB:-adb}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging
_log() {
    local color="$1" prefix="$2"
    shift 2
    echo -e "${color}[${prefix}]${NC} $*"
}

log_info()  { _log "$GREEN"  "INFO"  "$@"; }
log_warn()  { _log "$YELLOW" "WARN"  "$@"; }
log_error() { _log "$RED"    "ERROR" "$@"; }
log_ssh()   { _log "$BLUE"   "SSH"   "$@"; }
log_adb()   { _log "$CYAN"   "ADB"   "$@"; }

# Check if a local TCP port is open
port_is_open() {
    local host="${1:-localhost}" port="$2"
    if command -v nc &>/dev/null; then
        nc -z -w1 "$host" "$port" 2>/dev/null
    elif command -v bash &>/dev/null; then
        (echo >/dev/tcp/"$host"/"$port") 2>/dev/null
    elif command -v timeout &>/dev/null; then
        timeout 1 bash -c "echo >/dev/tcp/$host/$port" 2>/dev/null
    else
        return 1
    fi
}

# Check if ADB is available. Returns 0 if yes, 1 if no. Does not exit.
check_adb() {
    local is_wsl=false
    if grep -qi microsoft /proc/version 2>/dev/null; then
        is_wsl=true
    fi

    if [ "$is_wsl" = "true" ]; then
        command -v "$ADB" &>/dev/null && return 0
        return 1
    fi

    if command -v "$ADB" &>/dev/null; then
        return 0
    fi

    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            local candidates=(
                "/c/Users/$USERNAME/AppData/Local/Android/Sdk/platform-tools/adb.exe"
                "/c/Program Files/Android/platform-tools/adb.exe"
                "/c/Program Files/Meta Quest Developer Hub/resources/bin/adb.exe"
            )
            for candidate in "${candidates[@]}"; do
                if [ -x "$candidate" ]; then
                    ADB="$candidate"
                    export ADB
                    log_info "Using adb at: $ADB"
                    return 0
                fi
            done
            ;;
    esac

    return 1
}

# Ensure ADB is available
require_adb() {
    if check_adb; then
        return 0
    fi
    if grep -qi microsoft /proc/version 2>/dev/null; then
        log_error "ADB not found in WSL PATH. Install platform-tools inside WSL."
    else
        log_error "ADB not found. Please install Android SDK platform-tools."
    fi
    exit 1
}
