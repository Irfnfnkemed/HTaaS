import threading
from typing import List, Dict, Tuple


class GPU:
    def __init__(self, node_ip: str, device_id: int):
        self.node_ip = node_ip
        self.device_id = device_id
        self.available = True

    def __repr__(self):
        return f"{self.node_ip}:{self.device_id}"


class Node:
    def __init__(self, node_ip: str):
        self.node_ip = node_ip
        self.device_list: Dict[int, GPU] = {}
        self.available = 0

    def set_device(self, device_id: List[int]):
        for gpu_id in device_id:
            if gpu_id not in self.device_list:
                self.device_list[gpu_id] = GPU(self.node_ip, gpu_id)
                self.available += 1


class Allocator:
    def __init__(self):
        self._gpus: Dict[str, Node] = {}  # node_ip -> Node
        self._jobs_occupy: Dict[int, List[GPU]] = {}
        self._lock = threading.Lock()
        self._gpus_num = 0
        self._gpus_busy_num = 0

    def set_device(self, node_ip: str, device_id: List[int]):
        if node_ip not in self._gpus:
            self._gpus[node_ip] = Node(node_ip)
        self._gpus[node_ip].set_device(device_id)
        self._gpus_num += len(device_id)

    def alloca(self, job_id: int, num: int) -> List[str]:
        with self._lock:
            if job_id in self._jobs_occupy:
                for gpu in self._jobs_occupy[job_id]:  # Release occupied gpus
                    self._gpus[gpu.node_ip].available += 1
                    gpu.available = True
                    self._gpus_busy_num -= 1
            sorted_nodes = sorted(self._gpus.values(), key=lambda node: node.available, reverse=True)
            alloca_list = []
            for node in sorted_nodes:
                for gpu in node.device_list.values():
                    if gpu.available:
                        node.available -= 1
                        gpu.available = False
                        alloca_list.append(gpu)
                    if len(alloca_list) == num:
                        break
                if len(alloca_list) == num:
                    break
            self._jobs_occupy[job_id] = alloca_list
            result = [repr(gpu) for gpu in alloca_list]
            self._gpus_busy_num += len(result)
            return result

    def free(self, job_id: int):
        with self._lock:
            for gpu in self._jobs_occupy[job_id]:
                self._gpus[gpu.node_ip].available += 1
                gpu.available = True
                self._gpus_busy_num -= 1
            self._jobs_occupy.pop(job_id)

    def get_status(self) -> Tuple[int, int]:
        with self._lock:
            gpus_num = self._gpus_num
            gpus_busy_num = self._gpus_busy_num
        return gpus_num, gpus_busy_num
