#!/usr/bin/env python
"""Compose control 健康检查：RPC 端口已监听。"""
from __future__ import annotations

import os
import socket
import sys


def main() -> int:
    port = int(os.environ.get("DODO_RPC_HEALTH_PORT", "53151"))
    for host in ("127.0.0.1", socket.gethostname(), "control"):
        try:
            sock = socket.create_connection((host, port), 1)
            sock.close()
            return 0
        except OSError:
            continue
    return 1


if __name__ == "__main__":
    sys.exit(main())
