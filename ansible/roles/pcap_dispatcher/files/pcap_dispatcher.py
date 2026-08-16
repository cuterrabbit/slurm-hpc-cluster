#!/usr/bin/env python3

import datetime
import http.server
import json
import subprocess
import sys

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 9200
CAPTURE_INTERFACE = "enp0s8"
CAPTURE_SECONDS = 20
SSH_TIMEOUT_SECONDS = 5
NFS_DEST_DIR = "/apps/pcap-captures"
TARGET_ALERTNAME = "TetragonRetransmitSpike"
CAPTURE_REASON = "retransmit-spike"


def node_from_instance(instance):
    return instance.split(":")[0]


def ssh_run(node, remote_command, timeout):
    return subprocess.run(
        [
            "ssh",
            "-o", f"ConnectTimeout={SSH_TIMEOUT_SECONDS}",
            "-o", "StrictHostKeyChecking=no",
            node,
            remote_command,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def capture_on_node(node):
    epoch = int(datetime.datetime.now().timestamp())
    filename = f"{node}_{epoch}_{CAPTURE_REASON}.pcap"
    remote_tmp = f"/tmp/{filename}"
    nfs_path = f"{NFS_DEST_DIR}/{filename}"

    capture_cmd = f"sudo timeout {CAPTURE_SECONDS} tcpdump -i {CAPTURE_INTERFACE} -w {remote_tmp}"
    print(f"[{node}] capture start: {capture_cmd}", file=sys.stderr)
    capture_result = ssh_run(node, capture_cmd, timeout=CAPTURE_SECONDS + SSH_TIMEOUT_SECONDS + 5)
    print(
        f"[{node}] capture rc={capture_result.returncode} stderr={capture_result.stderr.strip()}",
        file=sys.stderr,
    )

    copy_cmd = f"sudo cp {remote_tmp} {nfs_path} && sudo chmod 644 {nfs_path} && echo copy-ok"
    copy_result = ssh_run(node, copy_cmd, timeout=SSH_TIMEOUT_SECONDS + 10)
    print(
        f"[{node}] copy rc={copy_result.returncode} stdout={copy_result.stdout.strip()} "
        f"stderr={copy_result.stderr.strip()}",
        file=sys.stderr,
    )

    cleanup_cmd = f"sudo rm -f {remote_tmp} && echo cleanup-ok"
    cleanup_result = ssh_run(node, cleanup_cmd, timeout=SSH_TIMEOUT_SECONDS + 5)
    print(
        f"[{node}] cleanup rc={cleanup_result.returncode} stdout={cleanup_result.stdout.strip()} "
        f"stderr={cleanup_result.stderr.strip()}",
        file=sys.stderr,
    )
    return nfs_path


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/capture":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return

        self.send_response(200)
        self.end_headers()

        for alert in payload.get("alerts", []):
            if alert.get("status") != "firing":
                continue
            labels = alert.get("labels", {})
            if labels.get("alertname") != TARGET_ALERTNAME:
                continue
            instance = labels.get("instance", "")
            if not instance:
                continue
            node = node_from_instance(instance)
            import threading

            threading.Thread(target=capture_on_node, args=(node,), daemon=True).start()

    def log_message(self, format, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))


def main():
    server = http.server.HTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    print(f"pcap dispatcher listening on {LISTEN_HOST}:{LISTEN_PORT}", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
