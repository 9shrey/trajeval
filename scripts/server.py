"""Start or stop one model server from experiments.yaml, tracked by a PID file.

    python scripts/server.py start qwen3b-awq     # returns once the server answers /v1/models
    python scripts/server.py stop qwen3b-awq      # stops exactly the process group it started

Stopping only ever signals the process group recorded at start, never a pattern match.
"""

import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reproduce import LOGS, start_server  # noqa: E402


def pid_file(name: str) -> Path:
    return LOGS / f"{name}.pid"


def main():
    action, name = sys.argv[1], sys.argv[2]
    if action == "start":
        if pid_file(name).exists():
            sys.exit(f"{name} already has a pid file; stop it first")
        p = start_server(name)
        pid_file(name).write_text(str(p.pid))
        print(f"{name} up, process group {p.pid}")
    elif action == "stop":
        pgid = int(pid_file(name).read_text())
        try:
            os.killpg(pgid, signal.SIGTERM)
            for _ in range(60):
                os.killpg(pgid, 0)
                time.sleep(1)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        pid_file(name).unlink()
        time.sleep(5)
        print(f"{name} stopped")
    else:
        sys.exit("usage: server.py start|stop <name>")


if __name__ == "__main__":
    main()
