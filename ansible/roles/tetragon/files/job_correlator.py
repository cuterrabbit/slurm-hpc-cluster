#!/usr/bin/env python3
"""Tetragon 이벤트에 Slurm job ID를 붙여서 Loki(promtail)로 넘길 로그 파일에 쓴다.

Tetragon 자체는 cgroup 경로를 주지 않는다(ADR-0019 실행 결과 참고 — Process 메시지에는
k8s용 pod/docker 필드만 있고 순수 cgroup 잡을 위한 필드가 없다). 대신 모든 이벤트에 pid는
항상 찍힌다.

1차 시도(주기적 cgroup.procs 폴링만으로 PID -> Job ID를 조인하는 방식)는 실측에서 진짜
버그를 드러냈다: process_exec 이벤트는 프로세스가 막 생성된 순간 나오는데, 그 시점엔 아직
다음 폴링 주기가 안 와서 매핑 테이블에 그 PID가 없다 — 그래서 정작 가장 중요한 "무슨 잡이
방금 이 걸 실행했다"는 이벤트가 계속 job_id=none으로 빠졌다(process_exit은 한참 뒤에 나와서
그때는 폴링이 따라잡아 정상 태깅됐다 — 이 어긋남으로 버그를 잡았다).

그래서 지금은 이벤트 스트림 자체에서 즉시 판단하는 방식을 우선한다:
1) 프로세스 자신이 Slurm의 배치 스크립트(`/var/spool/slurmd/job<N>/slurm_script`)면,
   경로에서 바로 job id를 뽑는다 — 폴링도 지연도 없다.
2) 아니면 부모(`parent_exec_id`)가 이미 알려진 job에 속해 있으면 그대로 물려받는다 —
   부모의 exec 이벤트는 자식의 exec 이벤트보다 항상 먼저 오므로 이것도 지연이 없다.
3) 그래도 못 찾으면(예: slurm_adopt로 트리 밖에서 잡에 편입된 프로세스), 그때만 폴백으로
   cgroup.procs 재귀 파싱(cgroup_exporter와 동일한 glob 패턴)을 쓴다 — 이 경로만 폴링
   지연이 남아있다.
"""
import glob
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

# exec_id -> job_id. 이벤트를 보는 순서대로 채워지므로 부모가 항상 자식보다 먼저 들어간다.
_exec_id_to_job = {}

# 폴백 전용: 주기적으로 갱신되는 PID -> Job ID (cgroup.procs 기반)
_pid_to_job = {}
_pid_to_job_lock = threading.Lock()


def build_pid_to_job_map():
    """cgroup_exporter와 동일한 glob 패턴으로 잡별 cgroup.procs를 재귀적으로 훑는다."""
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
                # 스크레이프 사이에 잡이 끝나서 cgroup이 사라지는 건 정상적인 레이스 컨디션이다.
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

    # 1) 자기 자신이 배치 스크립트인가 — 경로에 job id가 그대로 있다.
    m = JOB_SCRIPT_RE.search(binary)
    if m:
        job_id = m.group(1)
        if exec_id:
            _exec_id_to_job[exec_id] = job_id
        return job_id

    # 2) 부모가 이미 알려진 job에 속해 있는가.
    parent_exec_id = parent.get("exec_id") if parent else None
    if parent_exec_id and parent_exec_id in _exec_id_to_job:
        job_id = _exec_id_to_job[parent_exec_id]
        if exec_id:
            _exec_id_to_job[exec_id] = job_id
        return job_id

    # 3) 이미 스스로 캐시된 적 있는가(예: exit 이벤트가 exec보다 늦게 와서 재사용하는 경우).
    if exec_id and exec_id in _exec_id_to_job:
        return _exec_id_to_job[exec_id]

    # 4) 폴백 — slurm_adopt 등으로 부모 체인 밖에서 잡에 편입된 경우.
    pid = process.get("pid")
    if pid is not None:
        job_id = job_id_for_pid_fallback(pid)
        if job_id is not None:
            if exec_id:
                _exec_id_to_job[exec_id] = job_id
            return job_id

    return None


def forget_exec_id(process):
    """프로세스가 끝나면 캐시에서 지운다 — 안 지우면 무한정 자란다."""
    exec_id = process.get("exec_id")
    if exec_id:
        _exec_id_to_job.pop(exec_id, None)


def process_and_parent(event):
    for key in ("process_exec", "process_exit", "process_kprobe"):
        if key in event:
            payload = event[key]
            return payload.get("process", {}), payload.get("parent", {}), key
    return None, None, None


def wait_for_socket(path, timeout_seconds=60):
    """systemd의 After=/Requires=는 tetragon 프로세스가 떴다는 것만 보장하지,
    gRPC 소켓이 실제로 생성됐는지는 보장하지 않는다 — 그래서 직접 기다린다."""
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

    # 폴백 경로용 — 정상 케이스(배치 스크립트 + 그 자식들)는 이 스레드 없이도 즉시 태깅된다.
    refresher = threading.Thread(target=refresh_loop, daemon=True)
    refresher.start()

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
            event["slurm_job_id"] = job_id if job_id is not None else "none"

            if event_kind == "process_exit":
                forget_exec_id(process)

            out.write(json.dumps(event) + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
