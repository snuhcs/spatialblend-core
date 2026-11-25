# ADB Utilities

Forward a local ADB server to a remote Linux host over SSH `RemoteForward`.

## How it works

```
[Local machine]                                [Remote Linux server]
 adb server on localhost:5037                   ADB_SERVER_SOCKET=tcp:127.0.0.1:<remote_port>
            └────────SSH RemoteForward─────────→ localhost:<remote_port>
```

This forwards the ADB server itself (not device `adbd` TCP ports).

## Important

- Local role detection:
  - Windows shell (Git Bash/MSYS): client mode
  - WSL: client mode
  - Native Linux: server mode
- ADB selection:
  - WSL: uses WSL `adb` from PATH only
  - Windows Git Bash/MSYS: auto-detects Windows `adb.exe` if `adb` is not in PATH

## Quick Start

1. Local machine:
   `tools/adb/setup.sh --target c1`
2. If remote port `5037` is unavailable, script auto-falls back to an ephemeral remote port and prints:
   `export ADB_SERVER_SOCKET=tcp:127.0.0.1:<allocated_port>`
3. Remote server shell:
   `adb devices -l`

## Common local commands

- Fixed remote port:
  - `tools/adb/setup.sh --target c1 --port 5037`
- Start with remote ephemeral port:
  - `tools/adb/setup.sh --target c1 --auto-port`
- Keep Windows awake while tunnel is active:
  - `tools/adb/setup.sh --target c1 --prevent-sleep`
- Setup check only:
  - `tools/adb/setup.sh --target c1 --once`

## Options

- `--target HOST`: SSH host alias or destination (default: `c1`)
- `--port PORT`: preferred remote forwarded port (default: `5037`)
- `--auto-port`: request remote ephemeral port directly (`ssh -R 0:...`)
- `--no-auto-port-fallback`: disable automatic fallback when fixed remote port is busy
- `--once`: setup only, do not keep tunnel alive
- `--prevent-sleep`: prevent Windows sleep while tunnel is active
- `--no-sleep`: legacy alias for `--prevent-sleep`

## Troubleshooting

- `Could not resolve hostname ...`:
  - Fix `--target` or SSH config (`Host` entry).
- `remote port forwarding failed for listen port ...`:
  - Use `--auto-port` or another `--port`.
- `ADB not found in WSL PATH`:
  - Install platform-tools inside WSL.
