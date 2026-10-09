import base64
import hashlib
import os
import re

from .models import require


class ArtifactSink:
    def __init__(self, store, case_id, node, limit_mb):
        self.store, self.case_id, self.node = store, case_id, node
        self.directory = store.path / "raw" / case_id / node
        self.directory.mkdir(parents=True, exist_ok=True)
        self.files, self.warnings = {}, []
        self.collected = 0
        self.received, self.limit = 0, int(limit_mb * 1024 * 1024)

    def accept(self, event):
        kind = event["kind"]
        if kind == "artifact_warning":
            self.warnings.append(event)
            self.record(status="warning", **{key: value for key, value in event.items() if key != "kind"})
            return
        file_id = event.get("file_id", "")
        require(bool(re.fullmatch(r"[a-f0-9]{20}", file_id)), "Invalid artifact ID")
        if kind == "artifact_start":
            require(file_id not in self.files, "Duplicate artifact transfer ID")
            name = re.sub(r"[^A-Za-z0-9._-]", "_", event["name"])
            path = self.directory / (file_id + "-" + name[:100])
            temporary = path.with_name(path.name + ".partial")
            self.files[file_id] = {"handle": temporary.open("wb"), "temporary": temporary, "path": path,
                                   "sha": hashlib.sha256(), "bytes": 0, "source": event["source"]}
        elif kind == "artifact_chunk":
            item = self.files[file_id]
            chunk = base64.b64decode(event["data"], validate=True)
            require(self.received + len(chunk) <= self.limit, "Artifact transfer exceeds controller byte budget")
            item["handle"].write(chunk)
            item["sha"].update(chunk)
            item["bytes"] += len(chunk)
            self.received += len(chunk)
        elif kind == "artifact_end":
            item = self.files.pop(file_id)
            item["handle"].close()
            require(item["bytes"] == event["size"] and item["sha"].hexdigest() == event["sha256"], "Artifact checksum mismatch")
            os.replace(item["temporary"], item["path"])
            self.collected += 1
            self.record(status="collected", source=item["source"], path=str(item["path"]), size=item["bytes"], sha256=event["sha256"])

    def record(self, **fields):
        self.store.append("artifacts.jsonl", {"case_id": self.case_id, "node": self.node, **fields})

    def close(self):
        for item in self.files.values():
            item["handle"].close()
            self.warnings.append({"reason": "Incomplete artifact transfer", "source": item["source"]})
            self.record(status="incomplete", source=item["source"], path=str(item["temporary"]))
        self.files.clear()
