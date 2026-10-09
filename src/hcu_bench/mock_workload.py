import json
import sys
import time


def main():
    params = json.loads(sys.argv[1])
    print("MOCK ONLY: these values are not GPU/network benchmark measurements.", flush=True)
    for index in range(params["samples"]):
        time.sleep(params["delay_s"])
        if not params["empty"]:
            print("BENCH_RESULT " + json.dumps({"measurement_status": "simulated", "correctness": "not_checked",
                  "sample_id": index, "metrics": [{"name": "mock_payload", "value": params["payload_bytes"],
                  "unit": "bytes", "statistic": "configured"}]}), flush=True)
    return 7 if params["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
