"""AllReduce adapter for the rccl-tests ae3e635 output dialect, no launcher."""
import csv
import hashlib
import json
import math
import re


def integer(default, minimum, maximum, description, identity=True):
    return dict(type="integer", default=default, minimum=minimum, maximum=maximum, description=description, identity=identity)


class RcclAllReduce:
    protocol = "hcubench.external.v1"
    version = "0.1.0a3"
    description = "单机 AllReduce；使用者启动 MPI，插件导入原生日志。支持 ae3e635 输出格式。"
    schema = {
        "binary": dict(type="string", default="/work/rccl-tests/build/all_reduce_perf", description="rccl-tests 可执行文件绝对路径", identity=False, location=True),
        "ranks": integer(2, 2, 8, "单机 MPI 进程数，每进程一线程一卡"),
        "min_bytes": integer(8192, 8, 1073741824, "起始消息字节数"),
        "max_bytes": integer(67108864, 8, 1073741824, "结束消息字节数，必须落在倍增序列上"),
        "step_factor": integer(2, 2, 16, "消息大小倍增因子"),
        "datatype": dict(type="string", default="half", enum=["half", "float", "double"], description="数据类型", identity=True),
        "operation": dict(type="string", default="sum", enum=["sum"], description="归约操作，首版支持 sum", identity=True),
        "warmup_iters": integer(20, 1, 10000, "预热迭代次数", False),
        "iters": integer(50, 1, 100000, "测量迭代次数", False),
    }

    def sizes(self, p):
        result, size = [], p["min_bytes"]
        while size <= p["max_bytes"]:
            result.append(size)
            size *= p["step_factor"]
        return result

    def validate(self, p):
        if not p["binary"].startswith("/") or any(c in p["binary"] for c in ("\n", "\r", "\x00")) or "YOUR_" in p["binary"]:
            raise ValueError("binary must be an explicit Linux absolute path")
        sizes = self.sizes(p)
        if not sizes or sizes[-1] != p["max_bytes"]:
            raise ValueError("max_bytes must lie on the min_bytes * step_factor sequence")
        width = {"half": 2, "float": 4, "double": 8}[p["datatype"]]
        if any(s % width for s in sizes):
            raise ValueError("Message sizes must be divisible by dtype width")

    def preflight(self, p):
        return {"scope": "configuration_only", "runtime_checked": False,
                "requirements": ["MPI-enabled rccl-tests", "one idle HCU per MPI rank", "matching DTK/RCCL/MPI libraries", "capture stdout, stderr, returncode and runtime environment"]}

    def plan(self, p):
        test = [p["binary"], "-t", "1", "-g", "1", "-b", str(p["min_bytes"]), "-e", str(p["max_bytes"]), "-f", str(p["step_factor"]), "-d", p["datatype"], "-o", p["operation"], "-w", str(p["warmup_iters"]), "-n", str(p["iters"]), "-m", "1", "-c", "1", "-G", "0", "-a", "1"]
        return {"test_argv": test, "argv": ["mpirun", "--allow-run-as-root", "-np", str(p["ranks"])] + test,
                "expected_sizes_bytes": self.sizes(p), "sample_kind": "iteration_aggregate",
                "execution_owner": "user", "modes": ["out_of_place", "in_place"],
                "notes": ["No execution performed", "Default MPI command is for a single host", "Set HIP_VISIBLE_DEVICES explicitly in the external environment"]}

    def assess(self, bundle, p, config_hash):
        meta = json.loads((bundle / "execution.json").read_text(encoding="utf-8"))
        stdout = (bundle / "stdout.log").read_bytes()
        stderr = (bundle / "stderr.log").read_bytes()
        text = stdout.decode("utf-8", errors="replace").replace("\r\n", "\n")
        issues = []
        launch_error = bool(re.search(r"Launch params .*larger than launch bounds", text + stderr.decode("utf-8", errors="replace")))
        if launch_error:
            issues.append("runtime_kernel_launch_bounds_violation")
        if meta.get("schema_version") != 1:
            issues.append("execution_schema_mismatch")
        if meta.get("config_hash") != config_hash:
            issues.append("config_hash_mismatch")
        expected = self.plan(p)
        if meta.get("argv") != expected["argv"]:
            issues.append("execution_command_mismatch")
        for name, data in (("stdout.log", stdout), ("stderr.log", stderr)):
            if meta.get("sha256", {}).get(name) != hashlib.sha256(data).hexdigest():
                issues.append("artifact_hash_mismatch:" + name)
        code = meta.get("returncode")
        execution = "completed" if type(code) is int and code == 0 else "failed" if type(code) is int else "unknown"
        if execution != "completed":
            issues.append("execution_not_successful")
        env = meta.get("environment", {})
        if not isinstance(env, dict):
            env = {}
        for key in ("hostname", "binary_sha256", "mpi_version", "dtk_version", "rccl_library_sha256", "variables"):
            if not env.get(key):
                issues.append("missing_environment:" + key)
        for key in ("binary_sha256", "rccl_library_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(env.get(key, ""))):
                issues.append("invalid_environment_hash:" + key)
        variables = env.get("variables", {})
        if not isinstance(variables, dict):
            variables = {}
            issues.append("invalid_environment_variables")
        for key in ("NCCL_TESTS_SPLIT", "NCCL_TESTS_SPLIT_MASK", "NCCL_TESTS_DEVICE"):
            if variables.get(key):
                issues.append("unsupported_environment_override:" + key)
        if not meta.get("started_at") or not meta.get("finished_at"):
            issues.append("missing_execution_timestamps")

        header = re.findall(r"^# nThread (\d+) nGpus (\d+) minBytes (\d+) maxBytes (\d+) step: (\d+)\(factor\) warmup iters: (\d+) iters: (\d+) agg iters: (\d+) validation: (\d+) graph: (\d+)\s*$", text, re.M)
        wanted = (1, 1, p["min_bytes"], p["max_bytes"], p["step_factor"], p["warmup_iters"], p["iters"], 1, 1, 0)
        if len(header) != 1 or tuple(map(int, header[0])) != wanted:
            issues.append("runtime_parameters_missing_or_mismatch")
        devices = re.findall(r"^#\s+Rank\s+(\d+) Pid\s+\d+ on\s+(\S+) device\s+(\d+) \[([^\]]+)\] (.+)$", text, re.M)
        if (len(devices) != p["ranks"] or {int(d[0]) for d in devices} != set(range(p["ranks"]))
                or len({d[1] for d in devices}) != 1 or len({d[3] for d in devices}) != p["ranks"]):
            issues.append("rank_or_device_mapping_mismatch")
        env["observed_devices"] = [dict(rank=int(d[0]), host=d[1], visible_device=int(d[2]), pci_bus=d[3], name=d[4]) for d in devices]
        versions = re.findall(r"^rccl-tests: Version (.+)$", text, re.M)
        if len(versions) != 1:
            issues.append("missing_or_duplicate_rccl_tests_version")
        env["rccl_tests_reported_version"] = versions
        expected_columns = ["size", "count", "type", "redop", "root", "time", "algbw", "busbw", "#wrong", "time", "algbw", "busbw", "#wrong"]
        columns = [line.lstrip("# ").split() for line in text.splitlines() if re.match(r"^#\s+size\s+count\s", line)]
        if columns != [expected_columns] or "out-of-place" not in text or "in-place" not in text:
            issues.append("unsupported_table_header")
        if not re.search(r"^#\s+\(B\)\s+\(elements\)\s+\(us\)\s+\(GB/s\)\s+\(GB/s\)\s+\(us\)\s+\(GB/s\)\s+\(GB/s\)\s*$", text, re.M):
            issues.append("unsupported_units")
        rows = []
        width = {"half": 2, "float": 4, "double": 8}[p["datatype"]]
        for line_no, line in enumerate(text.splitlines(), 1):
            if not re.match(r"^\s*\d+\s", line):
                continue
            fields = line.split()
            try:
                if len(fields) != 13:
                    raise ValueError("column count")
                size, count = int(fields[0]), int(fields[1])
                if fields[2:5] != [p["datatype"], p["operation"], "-1"] or size != count * width:
                    raise ValueError("row semantics")
                pair = []
                for mode, start in (("out_of_place", 5), ("in_place", 9)):
                    time, alg, bus = map(float, fields[start:start+3])
                    wrong_value = float(fields[start+3].rstrip("*"))
                    if not all(math.isfinite(v) for v in (time, alg, bus, wrong_value)) or time <= 0 or min(alg, bus, wrong_value) < 0 or not wrong_value.is_integer():
                        raise ValueError("invalid measurement")
                    if "*" in fields[start+3]:
                        issues.append(f"validation_error_marker:line_{line_no}")
                    pair.append(dict(size_bytes=size, count=count, datatype=fields[2], operation=fields[3], mode=mode, time_us=time, algbw_GBps=alg, busbw_GBps=bus, wrong=int(wrong_value), source_line=line_no))
                rows.extend(pair)
            except (ValueError, OverflowError):
                issues.append(f"malformed_data_row:line_{line_no}")
        actual = [(r["size_bytes"], r["mode"]) for r in rows]
        wanted_rows = [(s, m) for s in self.sizes(p) for m in ("out_of_place", "in_place")]
        if actual != wanted_rows:
            issues.append("missing_duplicate_or_unexpected_rows")
        checks = re.findall(r"^# Out of bounds values\s*:\s*(\d+)\s+(OK|FAILED)\s*$", text, re.M)
        check_ok = checks == [("0", "OK")]
        if not check_ok:
            issues.append("correctness_footer_failed_or_missing")
        avg = re.findall(r"^# Avg bus bandwidth\s*:\s*(\S+)(?:\s+(OK|FAILED))?[ \t]*$", text, re.M)
        try:
            average = float(avg[0][0])
            if len(avg) != 1 or not math.isfinite(average) or average < 0 or avg[0][1] == "FAILED":
                raise ValueError()
        except (IndexError, ValueError):
            average = None
            issues.append("bandwidth_footer_failed_or_missing")
        wrong = sum(r["wrong"] for r in rows)
        if wrong:
            issues.append("nonzero_wrong_count")
        correctness = {"passed": bool(rows) and actual == wanted_rows and check_ok and wrong == 0 and bool(header) and header[0][-2] == "1" and not launch_error and not any("malformed" in i or "validation_error" in i for i in issues),
                       "reference": "rccl-tests built-in generated expected output", "policy": "-c 1; require all reported wrong counts and footer error count to be zero", "wrong_total": wrong}
        return {"execution": execution, "evidence_valid": not issues, "issues": issues, "environment": env,
                "correctness": correctness, "measurement": {"sample_kind": "iteration_aggregate", "unit": "us", "bandwidth_unit": "GB/s", "method": "rccl-tests wall-clock batch duration / iters / agg_iters, averaged across ranks (-a 1)",
                "boundary": "startColl loop through completeColl; validation outside timing; graph disabled", "iters": p["iters"], "warmup_iters": p["warmup_iters"], "raw_per_invocation_samples_available": False, "reported_average_busbw_GBps": average, "rows": rows}}

    def export_csv(self, measurement, path):
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["size_bytes", "count", "datatype", "operation", "mode", "time_us", "algbw_GBps", "busbw_GBps", "wrong", "source_line"])
            writer.writeheader()
            writer.writerows(measurement["rows"])
