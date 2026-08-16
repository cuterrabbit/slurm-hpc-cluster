#!/usr/bin/env python3

import glob
import http.server
import json
import os
import re
import subprocess
import sys
import threading
import time

JOBS_GLOB = "/sys/fs/cgroup/system.slice/*slurmstepd.scope/job_*"
JOB_SCRIPT_RE = re.compile(r"/var/spool/slurmd/job0*(\d+)/slurm_script")
REFRESH_INTERVAL_SECONDS = 2
TETRAGON_SOCKET = "unix:///var/run/tetragon/tetragon.sock"
TETRA_BIN = "/usr/local/bin/tetra"
OUTPUT_PATH = "/var/log/tetragon-enriched/events.json"
METRICS_PORT = 9799

_exec_id_to_job = {}

_pid_to_job = {}
_pid_to_job_lock = threading.Lock()

_retransmit_count = 0
_retransmit_count_lock = threading.Lock()


class MetricsHandler(http.server.BaseHTTPRequestHandler):


    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        with _retransmit_count_lock:
            count = _retransmit_count
        body = (
            "# HELP tetragon_tcp_retransmit_total Cumulative TCP retransmit events observed by Tetragon\n"
            "# TYPE tetragon_tcp_retransmit_total counter\n"
            f"tetragon_tcp_retransmit_total {count}\n"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def metrics_server_loop():
    server = http.server.HTTPServer(("0.0.0.0", METRICS_PORT), MetricsHandler)
    server.serve_forever()


def build_pid_to_job_map():
    mapping = {}
    for job_dir in glob.glob(JOBS_GLOB):
        job_id = os.path.basename(job_dir).split("_", 1)[-1]
        for root, _dirs, _files in os.walk(job_dir):
            try:
                with open(os.path.join(root, "cgroup.procs")) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            mapping[int(line)] = job_id
            except (FileNotFoundError, PermissionError, ValueError):
                continue
    return mapping


def refresh_loop():
    global _pid_to_job
    while True:
        new_map = build_pid_to_job_map()
        with _pid_to_job_lock:
            _pid_to_job = new_map
        time.sleep(REFRESH_INTERVAL_SECONDS)


def job_id_for_pid_fallback(pid):
    with _pid_to_job_lock:
        return _pid_to_job.get(pid)


def determine_job_id(process, parent):
    exec_id = process.get("exec_id")
    binary = process.get("binary", "") or ""

    m = JOB_SCRIPT_RE.search(binary)
    if m:
        job_id = m.group(1)
        if exec_id:
            _exec_id_to_job[exec_id] = job_id
        return job_id

    parent_exec_id = parent.get("exec_id") if parent else None
    if parent_exec_id and parent_exec_id in _exec_id_to_job:
        job_id = _exec_id_to_job[parent_exec_id]
        if exec_id:
            _exec_id_to_job[exec_id] = job_id
        return job_id

    
    if exec_id and exec_id in _exec_id_to_job:
        return _exec_id_to_job[exec_id]

    
    pid = process.get("pid")
    if pid is not None:
        job_id = job_id_for_pid_fallback(pid)
        if job_id is not None:
            if exec_id:
                _exec_id_to_job[exec_id] = job_id
            return job_id

    return None


def forget_exec_id(process):
    
    exec_id = process.get("exec_id")
    if exec_id:
        _exec_id_to_job.pop(exec_id, None)


def process_and_parent(event):
    for key in ("process_exec", "process_exit", "process_kprobe"):
        if key in event:
            payload = event[key]
            return payload.get("process", {}), payload.get("parent", {}), key
    return None, None, None


def build_summary(event_kind, payload, process, job_id):
    pid = process.get("pid")
    binary = process.get("binary") or "?"
    prefix = f"[job={job_id}]"

    if event_kind == "process_exec":
        return f"{prefix} exec {binary} (pid={pid})"
    if event_kind == "process_exit":
        return f"{prefix} exit {binary} (pid={pid})"
    if event_kind == "process_kprobe":
        function_name = payload.get("function_name") or "?"
        detail = ""
        for arg in payload.get("args", []):
            sock = arg.get("sock_arg")
            if sock:
                detail = (
                    f" {sock.get('saddr')}:{sock.get('sport')} -> "
                    f"{sock.get('daddr')}:{sock.get('dport')} [{sock.get('state')}]"
                )
                break
        return f"{prefix} {function_name} {binary} (pid={pid}){detail}"
    return f"{prefix} {event_kind}"


def wait_for_socket(path, timeout_seconds=60):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(1)
    return False


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    socket_path = TETRAGON_SOCKET.removeprefix("unix://")
    if not wait_for_socket(socket_path):
        print(f"tetragon socket {socket_path} not ready after timeout, exiting", file=sys.stderr)
        sys.exit(1)

    refresher = threading.Thread(target=refresh_loop, daemon=True)
    refresher.start()

    metrics_server = threading.Thread(target=metrics_server_loop, daemon=True)
    metrics_server.start()

    proc = subprocess.Popen(
        [TETRA_BIN, "getevents", "-o", "json", "--host", TETRAGON_SOCKET],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    with open(OUTPUT_PATH, "a", buffering=1) as out:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            process, parent, event_kind = process_and_parent(event)
            if process is None:
                out.write(json.dumps(event) + "\n")
                continue

            job_id = determine_job_id(process, parent)
            job_id_str = job_id if job_id is not None else "none"
            event["slurm_job_id"] = job_id_str
            event["summary"] = build_summary(event_kind, event[event_kind], process, job_id_str)

            if event_kind == "process_kprobe" and event[event_kind].get("function_name") == "tcp_retransmit_skb":
                global _retransmit_count
                with _retransmit_count_lock:
                    _retransmit_count += 1

            if event_kind == "process_exit":
                forget_exec_id(process)

            out.write(json.dumps(event) + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
