# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import sys

from .base import AGENT_PATH, AgentExecutor


class LocalExecutor(AgentExecutor):
    def launch_argv(self, spec):
        return [self.node.get("python", sys.executable), "-u", str(AGENT_PATH)]
