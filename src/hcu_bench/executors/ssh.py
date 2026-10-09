import shlex

from .base import AGENT_PATH, AgentExecutor


class SSHExecutor(AgentExecutor):
    def launch_argv(self, spec):
        node = self.node
        command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
                   "-o", f"ConnectTimeout={int(node.get('connect_timeout_s', 10))}", "-p", str(node.get("port", 22))]
        if node.get("identity_file"):
            command += ["-i", node["identity_file"]]
        if node.get("user"):
            command += ["-l", node["user"]]
        remote = shlex.join([node.get("python", "python3"), "-u", "-c", AGENT_PATH.read_text(encoding="utf-8")])
        return command + [node["host"], remote]
