from .local import LocalExecutor
from .ssh import SSHExecutor


def for_node(node):
    return LocalExecutor(node) if node["executor"] == "local" else SSHExecutor(node)
