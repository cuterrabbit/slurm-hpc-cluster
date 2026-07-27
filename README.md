## 구성

| 노드 | 역할 | 스펙 | 호스트 전용 IP |
|---|---|---|---|
| head | slurmctld, slurmdbd, MySQL, NFS, 모니터링 | 6GB RAM / 2vCPU+ / 50GB | 192.168.56.10 |
| node1 | slurmd (워커) | 4GB RAM / 2vCPU+ / 25GB | 192.168.56.11 |
| node2 | slurmd (워커) | 4GB RAM / 2vCPU+ / 25GB | 192.168.56.12 |

게스트 OS: Ubuntu 26.04 LTS, 노드 간 통신은 TCP.
