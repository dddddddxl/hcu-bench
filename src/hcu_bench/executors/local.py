import sys

from .base import AGENT_PATH, AgentExecutor


class LocalExecutor(AgentExecutor):
    def launch_argv(self, spec):
        return [self.node.get("python", sys.executable), "-u", str(AGENT_PATH)]
