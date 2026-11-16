# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from .local import LocalExecutor
from .ssh import SSHExecutor


def for_node(node):
    return LocalExecutor(node) if node["executor"] == "local" else SSHExecutor(node)
