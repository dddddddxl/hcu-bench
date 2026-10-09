import threading
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed

from .adapters import describe_spec, for_case
from .artifacts import ArtifactSink
from .executors import for_node
from .models import BenchError, Plan
from .results import parse_sample
from .store import RunStore, now

GOOD = {"completed", "duration_reached"}
FINAL_FAILURE = {"failed", "timed_out", "invalid", "cleanup_unconfirmed"}


def run_plan(plan: Plan, store: RunStore, quiet=False) -> int:
    store.state(status="running", started_at=now())
    aborted = False
    cancelled = threading.Event()
    old_signals = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_signals[signum] = signal.signal(signum, lambda *_: cancelled.set())
    try:
        for case in plan.cases:
            if store.stopped() or cancelled.is_set():
                store.case(case.id, status="cancelled", reason="Run cancellation requested")
                continue
            if aborted:
                store.case(case.id, status="not_run", reason="Earlier failure stopped the plan")
                continue
            store.case(case.id, status="running", started_at=now())
            peer_failed = threading.Event()
            results = {}

            def execute_node(node_id):
                valid = 0
                invalid = 0
                correctness_failed = False
                logs = store.path / "logs" / case.id
                logs.mkdir(exist_ok=True)
                raw_path = logs / f"{node_id}.raw.log"
                timed_path = logs / f"{node_id}.log"
                sink = ArtifactSink(store, case.id, node_id, case.pack.get("artifact_limit_mb", 100))
                with raw_path.open("w", encoding="utf-8") as raw, timed_path.open("w", encoding="utf-8") as timed:
                    def on_event(event):
                        nonlocal valid, invalid, correctness_failed
                        if event.get("kind", "").startswith("artifact_"):
                            try:
                                sink.accept(event)
                            except Exception as exc:
                                sink.warnings.append({"reason": str(exc)})
                                sink.record(status="warning", reason=str(exc))
                            if event["kind"] == "artifact_warning":
                                warning = f"[{now()}] [archive warning] {event.get('source', '')}: {event.get('reason', '')}"
                                timed.write(warning + "\n")
                                timed.flush()
                                if not quiet:
                                    print(warning, flush=True)
                            return
                        if event.get("kind") in ("environment", "container_environment"):
                            store.environment(f"{case.id}/{node_id}/{event['kind']}", event.get("data", {}))
                            return
                        if event.get("kind") != "log":
                            return
                        text = event.get("text", "")
                        timestamp = event.get("timestamp", now())
                        raw.write(text + "\n")
                        raw.flush()
                        timed.write(f"[{timestamp}] {text}\n")
                        timed.flush()
                        if not quiet:
                            print(f"[{timestamp}] [{case.test_id}/{node_id}] {text}", flush=True)
                        if case.pack["result_protocol"] == "none":
                            return
                        try:
                            sample = parse_sample(text)
                        except BenchError as exc:
                            sample = {"measurement_status": "invalid", "correctness": "not_checked", "metrics": [], "reason": str(exc)}
                        if sample is None:
                            return
                        if case.adapter != "mock" and sample["measurement_status"] == "simulated":
                            sample = {"measurement_status": "invalid", "correctness": "not_checked", "metrics": [], "reason": "Real test emitted a simulated result"}
                        if sample["measurement_status"] in ("measured", "simulated"):
                            valid += 1
                        elif sample["measurement_status"] == "invalid":
                            invalid += 1
                            timed.write(f"[{timestamp}] [warning] Invalid sample skipped: {sample.get('reason', 'not supplied')}\n")
                            timed.flush()
                        correctness_failed |= sample["correctness"] == "failed"
                        store.append("results.jsonl", {**sample, "run_id": store.path.name, "case_id": case.id, "test_id": case.test_id,
                                     "suite": case.suite, "node": node_id, "node_rank": case.nodes.index(node_id), "params": case.params,
                                     "repeat": case.repeat, "observed_at": timestamp, "simulated": case.adapter == "mock"})

                    try:
                        spec = for_case(case).build(plan, case, node_id, store.path)
                        store.append("commands.jsonl", {"case_id": case.id, "node": node_id, "spec": describe_spec(spec)})
                        result = for_node(plan.nodes[node_id]).execute(spec, lambda: peer_failed.is_set() or cancelled.is_set() or store.stopped(), on_event)
                    except Exception as exc:
                        result = {"status": "failed", "cleanup": "unconfirmed", "error": str(exc)}
                    finally:
                        sink.close()
                    if case.pack.get("artifacts") and not sink.collected and not sink.warnings:
                        sink.warnings.append({"reason": "No declared report was received"})
                        sink.record(status="warning", reason="No declared report was received")
                    if result.get("cleanup") != "confirmed":
                        result["status"] = "cleanup_unconfirmed"
                    elif correctness_failed:
                        result["status"] = "failed"
                        result["error"] = "Correctness failure was reported"
                    elif result["status"] in GOOD and case.pack["requires_metrics"] and valid == 0:
                        result["status"] = "invalid"
                        result["error"] = "No valid measurement samples; process exit alone is not a performance result"
                    result.update(valid_samples=valid, invalid_samples=invalid, raw_log=str(raw_path), log=str(timed_path),
                                  artifact_status="warnings" if sink.warnings else "collected" if case.pack.get("artifacts") else "not_requested")
                    if result["status"] not in GOOD:
                        peer_failed.set()
                    return result

            with ThreadPoolExecutor(max_workers=len(case.nodes)) as pool:
                futures = {pool.submit(execute_node, node): node for node in case.nodes}
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
            statuses = {value["status"] for value in results.values()}
            if "cleanup_unconfirmed" in statuses:
                status = "cleanup_unconfirmed"
                aborted = True
            elif store.stopped() or cancelled.is_set():
                status = "cancelled"
            elif not statuses <= GOOD:
                status = next((value for value in ("failed", "timed_out", "invalid", "cancelled") if value in statuses), "failed")
            else:
                status = "duration_reached" if "duration_reached" in statuses else "completed"
            store.case(case.id, status=status, finished_at=now(), nodes=results)
            if status in FINAL_FAILURE and plan.failure_policy == "stop":
                aborted = True
    except KeyboardInterrupt:
        cancelled.set()
        store.state(status="cancelled", finished_at=now())
        return 130
    except Exception as exc:
        store.state(status="failed", error=str(exc), finished_at=now())
        raise
    finally:
        for signum, handler in old_signals.items():
            signal.signal(signum, handler)
    states = [value["status"] for value in store.read("state.json")["cases"].values()]
    if store.stopped() or cancelled.is_set():
        final = "cancelled"
    elif any(state in FINAL_FAILURE or state == "not_run" for state in states):
        final = "failed"
    elif "cancelled" in states:
        final = "cancelled"
    else:
        final = "completed"
    store.state(status=final, finished_at=now())
    return 130 if final == "cancelled" else 1 if final == "failed" else 0
