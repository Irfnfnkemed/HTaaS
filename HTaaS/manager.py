from dataclasses import dataclass
import threading
from typing import List, Dict, Tuple


"""
This module manages GPU resources and allocates them to jobs.
"""


@dataclass
class GPU:

    node_ip: str
    device_id: int
    available: bool = True

    def __repr__(self):
        return f"{self.node_ip}:{self.device_id}"


@dataclass
class Node:

    node_ip: str
    device_ids: Dict[int, GPU] = {}
    available: int = 0

    def set_device(self, device_ids: List[int]):
        """
        Set GPUs for the node.

        Args:
        device_ids: The list of device ids of the GPUs on the node.
        """
        for gpu_id in device_ids:
            if gpu_id not in self.device_ids:
                self.device_ids[gpu_id] = GPU(self.node_ip, gpu_id)
                self.available += 1


@dataclass
class Allocator:

    nodes: Dict[str, Node] = {}  # node_ip -> Node
    jobs_occupy: Dict[int, List[GPU]] = {}
    lock: threading.Lock = threading.Lock()
    gpus_num: int = 0
    gpus_busy_num: int = 0

    def set_device(self, node_ip: str, device_ids: List[int]):
        """
        Set GPUs for the node.

        Args:
        node_ip: The IP address of the node.
        device_ids: The list of device ids of the GPUs on the node.
        """
        if node_ip not in self.nodes:
            self.nodes[node_ip] = Node(node_ip)
        self.nodes[node_ip].set_device(device_ids)
        self.gpus_num += len(device_ids)

    def alloca(self, job_id: int, num: int) -> List[str]:
        """
        Allocate GPUs to the job.

        Args:
        job_id: The job id.
        num: The number of GPUs to allocate.

        Returns:
        A list of allocated GPU identifiers.
        """
        with self.lock:
            if job_id in self.jobs_occupy:
                for gpu in self.jobs_occupy[job_id]:  # Release occupied gpus
                    self.nodes[gpu.node_ip].available += 1
                    gpu.available = True
                    self.gpus_busy_num -= 1
            sorted_nodes = sorted(
                self.nodes.values(), key=lambda node: node.available, reverse=True
            )
            alloca_list = []
            for node in sorted_nodes:
                for gpu in node.device_ids.values():
                    if gpu.available:
                        node.available -= 1
                        gpu.available = False
                        alloca_list.append(gpu)
                    if len(alloca_list) == num:
                        break
                if len(alloca_list) == num:
                    break
            self.jobs_occupy[job_id] = alloca_list
            result = [repr(gpu) for gpu in alloca_list]
            self.gpus_busy_num += len(result)
            return result

    def free(self, job_id: int):
        """
        Free allocated GPUs for the job.

        Args:
        job_id: The job id.
        """
        with self.lock:
            for gpu in self.jobs_occupy[job_id]:
                self.nodes[gpu.node_ip].available += 1
                gpu.available = True
                self.gpus_busy_num -= 1
            self.jobs_occupy.pop(job_id)

    def get_status(self) -> Tuple[int, int]:
        """
        Get the current status of the GPU resources.

        Returns:
        A tuple of two integers. The first element is the total number of GPUs, and the second element is the number of GPUs that are currently busy.
        """
        with self.lock:
            gpus_num = self.gpus_num
            gpus_busy_num = self.gpus_busy_num
        return gpus_num, gpus_busy_num
