#!/bin/bash
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

# ADB Remote Setup (ADB server forwarding)
#
# OS-aware ADB bridge:
#   Windows/macOS: ensure local adb server + keep SSH reverse tunnel alive
#   Linux:         use forwarded adb server via ADB_SERVER_SOCKET
#
# Usage:
#   setup.sh
#   setup.sh --once
#   setup.sh --target c1
#   setup.sh --port 5037

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

ADB_SERVER_PORT="${SB_ADB_SERVER_PORT:-5037}"
SSH_TARGET="${SB_ADB_SSH_TARGET:-c1}"
ONCE_MODE=false
PREVENT_SLEEP=false
AUTO_PORT_MODE=false
AUTO_PORT_FALLBACK=true

SSH_PID=""
CAFFEINE_PID=""
SSH_ERR_FILE=""
STOP_REQUESTED=false
FORWARDED_PORT=""

# --- OS detection ---
detect_os() {
    case "$(uname -s)" in
        Linux*)
            if grep -qi microsoft /proc/version 2>/dev/null; then
                echo "client"
            else
                echo "server"
            fi
            ;;
        *)       echo "client" ;;
    esac
}

OS_ROLE="$(detect_os)"

# --- Windows sleep guard ---
prevent_sleep_start() {
    local is_windows_shell=false
    local is_wsl=false

    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*) is_windows_shell=true ;;
    esac
    if grep -qi microsoft /proc/version 2>/dev/null; then
        is_wsl=true
    fi
    if [ "$is_windows_shell" != "true" ] && [ "$is_wsl" != "true" ]; then
        return
    fi

    log_info "Preventing Windows sleep..."
    powershell.exe -NoProfile -Command '
        $code = @"
        [DllImport("kernel32.dll")]
        public static extern uint SetThreadExecutionState(uint esFlags);
"@
        $ES = Add-Type -MemberDefinition $code -Name "WinAPI" -PassThru
        $ES::SetThreadExecutionState(0x80000001) | Out-Null
        while ($true) {
            Start-Sleep -Seconds 60
            $ES::SetThreadExecutionState(0x80000001) | Out-Null
        }
    ' &>/dev/null &
    CAFFEINE_PID=$!
}

prevent_sleep_stop() {
    if [ -n "$CAFFEINE_PID" ] && kill -0 "$CAFFEINE_PID" 2>/dev/null; then
        kill "$CAFFEINE_PID" 2>/dev/null || true
    fi
}

# --- tunnel lifecycle ---
tunnel_cleanup() {
    if [ -n "$SSH_PID" ] && kill -0 "$SSH_PID" 2>/dev/null; then
        log_info "Stopping SSH tunnel..."
        kill "$SSH_PID" 2>/dev/null || true
        wait "$SSH_PID" 2>/dev/null || true
    fi
    prevent_sleep_stop
    if [ -n "$SSH_ERR_FILE" ] && [ -f "$SSH_ERR_FILE" ]; then
        rm -f "$SSH_ERR_FILE" 2>/dev/null || true
    fi
    log_info "Done."
}

start_adb_server() {
    require_adb

    if "$ADB" start-server >/dev/null 2>&1; then
        log_adb "Local adb server ready on port $ADB_SERVER_PORT"
    else
        log_error "Failed to start local adb server."
        exit 1
    fi
}

keep_tunnel_alive() {
    trap tunnel_cleanup EXIT
    trap 'STOP_REQUESTED=true' INT TERM HUP

    if [ "$PREVENT_SLEEP" = "true" ]; then
        prevent_sleep_start
    fi

    log_info "Keeping SSH tunnel alive to $SSH_TARGET (Ctrl+C to stop)"
    if [ "$AUTO_PORT_MODE" = "true" ]; then
        log_info "  $SSH_TARGET:(auto) -> localhost:$ADB_SERVER_PORT"
    else
        log_info "  $SSH_TARGET:$ADB_SERVER_PORT -> localhost:$ADB_SERVER_PORT"
    fi
    echo ""

    SSH_ERR_FILE=$(mktemp 2>/dev/null || echo "/tmp/sb-adb-ssh.err")
    : > "$SSH_ERR_FILE"

    while true; do
        if [ "$STOP_REQUESTED" = "true" ]; then
            break
        fi

        if [ -n "$SSH_PID" ] && kill -0 "$SSH_PID" 2>/dev/null; then
            sleep 10
            continue
        fi

        [ -n "$SSH_PID" ] && log_warn "Tunnel died. Reconnecting..."

        local remote_port="$ADB_SERVER_PORT"
        local ssh_verbose_opt=()
        if [ "$AUTO_PORT_MODE" = "true" ]; then
            remote_port="0"
            ssh_verbose_opt=(-v)
        fi

        ssh -N \
            -o ServerAliveInterval=10 \
            -o ServerAliveCountMax=3 \
            -o ExitOnForwardFailure=yes \
            -o TCPKeepAlive=yes \
            -o ConnectTimeout=10 \
            "${ssh_verbose_opt[@]}" \
            -R "$remote_port:127.0.0.1:$ADB_SERVER_PORT" \
            "$SSH_TARGET" \
            2>"$SSH_ERR_FILE" &
        SSH_PID=$!

        sleep 3
        if [ "$STOP_REQUESTED" = "true" ]; then
            break
        fi

        if kill -0 "$SSH_PID" 2>/dev/null; then
            log_ssh "Tunnel up (PID: $SSH_PID)"
            if [ "$AUTO_PORT_MODE" = "true" ]; then
                local allocated=""
                allocated=$(grep -oE "Allocated port [0-9]+" "$SSH_ERR_FILE" 2>/dev/null | tail -n 1 | awk '{print $3}')
                if [ -n "$allocated" ]; then
                    FORWARDED_PORT="$allocated"
                    log_info "Remote port allocated: $FORWARDED_PORT"
                    log_info "Server: export ADB_SERVER_SOCKET=tcp:127.0.0.1:$FORWARDED_PORT"
                else
                    log_warn "Tunnel is up but allocated remote port was not detected in ssh output."
                fi
            else
                FORWARDED_PORT="$ADB_SERVER_PORT"
            fi
        else
            local ssh_err=""
            if [ -n "$SSH_ERR_FILE" ] && [ -f "$SSH_ERR_FILE" ]; then
                ssh_err=$(tail -n 1 "$SSH_ERR_FILE" 2>/dev/null | tr -d '\r')
            fi
            if [ -n "$ssh_err" ]; then
                log_warn "SSH error: $ssh_err"
            fi

            if [ -n "$SSH_ERR_FILE" ] && [ -f "$SSH_ERR_FILE" ] && grep -qiE "Could not resolve hostname|Name or service not known|Temporary failure in name resolution" "$SSH_ERR_FILE"; then
                log_error "SSH target '$SSH_TARGET' is not resolvable. Fix host/config and retry."
                return 1
            fi
            if [ -n "$SSH_ERR_FILE" ] && [ -f "$SSH_ERR_FILE" ] && grep -qiE "Permission denied|Host key verification failed|REMOTE HOST IDENTIFICATION HAS CHANGED|enter passphrase for key|no such identity|Too many authentication failures" "$SSH_ERR_FILE"; then
                log_error "SSH authentication/host-key failure for '$SSH_TARGET'. Fix SSH access and retry."
                return 1
            fi
            if [ -n "$SSH_ERR_FILE" ] && [ -f "$SSH_ERR_FILE" ] && grep -qiE "remote port forwarding failed for listen port" "$SSH_ERR_FILE"; then
                if [ "$AUTO_PORT_FALLBACK" = "true" ] && [ "$AUTO_PORT_MODE" != "true" ]; then
                    log_warn "Remote port $ADB_SERVER_PORT is unavailable. Switching to auto remote port."
                    AUTO_PORT_MODE=true
                    continue
                fi
            fi
            log_warn "Tunnel failed. Retrying in 10s..."
            SSH_PID=""
            sleep 10
        fi
    done
}

# --- client/server entries ---
client_main() {
    start_adb_server

    if [ "$ONCE_MODE" = "true" ]; then
        log_info "Setup complete. Run without --once to keep the tunnel alive."
        log_info "SSH command: ssh -N -R $ADB_SERVER_PORT:127.0.0.1:$ADB_SERVER_PORT $SSH_TARGET"
        exit 0
    fi

    keep_tunnel_alive
}

server_main() {
    require_adb

    if ! port_is_open "127.0.0.1" "$ADB_SERVER_PORT"; then
        log_info "Port $ADB_SERVER_PORT not open. No adb server forwarded yet."
        exit 0
    fi

    log_info "Forwarded adb server detected at 127.0.0.1:$ADB_SERVER_PORT"
    echo ""
    echo "Run this in your shell:"
    echo "  export ADB_SERVER_SOCKET=tcp:127.0.0.1:$ADB_SERVER_PORT"
    echo "  adb devices -l"
    echo ""

    local result
    result=$(ADB_SERVER_SOCKET="tcp:127.0.0.1:$ADB_SERVER_PORT" "$ADB" devices -l 2>&1 || true)
    echo "$result"
}

# --- main ---
main() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --once)      ONCE_MODE=true; shift ;;
            --prevent-sleep|--no-sleep) PREVENT_SLEEP=true; shift ;;
            --no-prevent-sleep) PREVENT_SLEEP=false; shift ;;
            --auto-port) AUTO_PORT_MODE=true; shift ;;
            --no-auto-port-fallback) AUTO_PORT_FALLBACK=false; shift ;;
            --target)    SSH_TARGET="$2"; shift 2 ;;
            --port)      ADB_SERVER_PORT="$2"; shift 2 ;;
            -h|--help)
                echo "Usage: setup.sh [--once] [--prevent-sleep] [--auto-port] [--target HOST] [--port PORT]"
                echo "  Windows/macOS: start adb server + keep SSH tunnel alive"
                echo "  Linux: use forwarded adb server via ADB_SERVER_SOCKET"
                echo "  --once:     setup only, don't keep tunnel"
                echo "  --prevent-sleep: prevent Windows from sleeping while tunnel is active"
                echo "  --no-sleep: legacy alias for --prevent-sleep"
                echo "  --auto-port: request remote ephemeral port (ssh -R 0:...)"
                echo "  --no-auto-port-fallback: disable automatic fallback when fixed remote port is busy"
                exit 0
                ;;
            *) shift ;;
        esac
    done

    if ! check_adb; then
        log_info "ADB not found. Skipping setup."
        exit 0
    fi

    if [ "$OS_ROLE" = "client" ]; then
        client_main
    else
        server_main
    fi
}

main "$@"
