"""CPU-only OpenAI fixture, not a model or a performance reference."""
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = {"data": [{"id": "fixture-model", "root": os.environ.get("FIXTURE_TOKENIZER", "")}]}
        self.send_response(200 if self.path in ("/health", "/v1/models") else 404)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if os.environ.get("FIXTURE_FAIL"):
            self.send_response(500)
            self.end_headers()
            return
        count = body.get("max_tokens", 8)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            for index in range(count):
                time.sleep(0.002)
                chunk = {"id": "fixture", "object": "chat.completion.chunk", "created": int(time.time()),
                         "model": "fixture-model", "choices": [{"index": 0, "delta": {"content": "hello "}, "finish_reason": None}]}
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.flush()
            final = {"id": "fixture", "object": "chat.completion.chunk", "created": int(time.time()),
                     "model": "fixture-model", "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                     "usage": {"prompt_tokens": 16, "completion_tokens": count, "total_tokens": 16 + count}}
            self.wfile.write(("data: " + json.dumps(final) + "\n\ndata: [DONE]\n\n").encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    ThreadingHTTPServer((os.environ["BENCH_HOST"], int(os.environ["BENCH_PORT"])), Handler).serve_forever()
