import math
import os
import copy
import threading
import time
import threading
from typing import Dict, List, Literal, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from .comm import *
from .config import *

"""
This module provides the Engine class, which schedules all the jobs.
"""

"""
This module provides the Engine class, which schedules all the jobs.
"""


@dataclass
class GPU:
    hash_id: int  # unique id in the
    node_ip: str
    device_id: int
    job_id: int = -1  # -1 means not allocated

    def __repr__(self):
        return f"{self.node_ip}:{self.device_id}"


class Allocator:
    gpus: Dict[int, GPU]  # gpu hash id -> GPU info
    jobs: Dict[int, List[int]]  # job id -> list of hash ids of the job gpus
    lock: threading.Lock

    def __init__(self):
        self.gpus = {}
        self.jobs = {}
        self.lock = threading.Lock()

    def add_device(self, node_ip: str, device_ids: List[int]):
        """
        Add GPUs.

        Args:
        node_ip: The IP address of the node.
        device_ids: The list of device ids of the GPUs on the node.
        """
        with self.lock:
            for device_id in device_ids:
                hash_id = hash(f"{node_ip}:{device_id}")
                assert hash_id not in self.gpus
                self.gpus[hash_id] = GPU(hash_id, node_ip, device_id)

    def alloca_job(self, job_id: int, num: int) -> List[str]:
        """
        Allocate GPUs to the job.

        Args:
        job_id: The job id.
        num: The number of GPUs to allocate.

        Returns:
        A list of allocated GPU identifiers.
        """
        with self.lock:
            nodes: Dict[str, List[int]] = defaultdict(list)
            available = 0
            for gpu in self.gpus.values():
                if gpu.job_id == -1:
                    nodes[gpu.node_ip].append(gpu.hash_id)
                    available += 1
            alloca_list = []
            if available == 0:
                # no other available gpus
                for hash_id in self.jobs[job_id]:
                    alloca_list.append(hash_id)
            else:
                # first release occupied gpus if existed
                if job_id in self.jobs:
                    for hash_id in self.jobs[job_id]:
                        gpu = self.gpus[hash_id]
                        gpu.job_id = -1
                        nodes[gpu.node_ip].append(gpu.hash_id)
                        available += 1
                    self.jobs[job_id].clear()
                else:
                    self.jobs[job_id] = []
                sorted_nodes = sorted(nodes.keys(), key=lambda k: len(nodes[k]), reverse=False)
                # try to allocate gpu in the same node
                for node_ip in sorted_nodes:
                    if len(nodes[node_ip]) >= num:
                        for hash_id in nodes[node_ip]:
                            assert self.gpus[hash_id].job_id == -1
                            self.gpus[hash_id].job_id = job_id
                            alloca_list.append(hash_id)
                            if len(alloca_list) == num:
                                break
                    if len(alloca_list) == num:
                        break
                assert len(alloca_list) == 0 or len(alloca_list) == num
                # try to allocate gpu in multiple nodes
                if len(alloca_list) == 0:
                    for node_ip in sorted_nodes:
                        for hash_id in nodes[node_ip]:
                            assert self.gpus[hash_id].job_id == -1
                            self.gpus[hash_id].job_id = job_id
                            alloca_list.append(hash_id)
                            if len(alloca_list) == num:
                                break
                        if len(alloca_list) == num:
                            break
                self.jobs[job_id] = copy.deepcopy(alloca_list)
                result = [repr(self.gpus[hash_id]) for hash_id in alloca_list]
        return result

    def free_job(self, job_id: int):
        with self.lock:
            for hash_id in self.jobs[job_id]:
                self.gpus[hash_id].job_id = -1
            self.jobs.pop(job_id)
            
    @property
    def total_gpu_num(self):
        with self.lock:
            return len(self.gpus)
            
    


class Job:
    _conn_engine: ClientInstance 
    _conn_job: Server
    _process_list: List[subprocess.Popen]
    
    
    def __init__(self):
        self._conn_engine = ClientInstance()
        self._conn_job = Server()
        self._process_list = []
        
    def run(
        self,
        engine_ip: str,
        engine_port: int,
        script_path: str,
        args: List[str],
        prev_cmd: Optional[List[str]] = None,
    ):
        """
        Launches, run and monitor the job.
        """

        # connect to the engine and prepare the job
        self._conn_engine.connect(engine_ip, engine_port)
        self._conn_engine.send("init", "")
        cmd, job_id = self._conn_engine.recv()
        print(f"JOB_ID:{job_id}...")
        os.makedirs(f"{WORK_DIR}/job_{job_id}", exist_ok=True)
        assert cmd == "init"
        # alloca 1 gpu to profile the job
        self._conn_engine.send("alloc", 1)
        cmd, gpu_list = self._conn_engine.recv()
        if cmd == "pending":
            while True:
                cmd, data = self._conn_engine.recv()
                if cmd == "pending":
                    if int(data) > MAX_PENDING_TIMES:
                        raise RuntimeError()
                elif cmd == "alloc":
                    gpu_list = data
                    break
        self._conn_job.serve(get_ip(), get_free_port())
        # serving loops
        # loop when the gpu-number changes (save checkpoint and restart)
        # end when the job finishes
        while True:
            # get the gpu info from the engine
            parsed_list = {}
            for entry in gpu_list:
                ip, gpu_id = entry.split(":")
                if ip in parsed_list:
                    parsed_list[ip].append(gpu_id)
                else:
                    parsed_list[ip] = [gpu_id]
            parsed_gpu_list = [(ip, parsed_list[ip]) for ip, ids in parsed_list.items()]
            # launch the job on each node
            self._process_list.clear()
            master_port = get_free_port(parsed_gpu_list[0][0], True)
            job_host = self._conn_job.get_ip()
            job_port = self._conn_job.get_port()
            pwd = script_path[: script_path.rfind("/")]
            env_set_cmd = [
                f"source {CONDA_ACTIVATE_ADDR} {CONDA_ENV_NAME}",
                f"export PROCESSOR_HOST={job_host}",
                f"export PROCESSOR_PORT={job_port}",
                f"export PYTHONPATH={os.environ['PYTHONPATH']}",
                f"cd {pwd}",
            ]
            env_set_cmd = " && ".join(env_set_cmd)
            for index, (ip, gpus) in enumerate(parsed_gpu_list):
                torchrun_cmd = [
                    "torchrun",
                    f"--nproc_per_node={len(gpus)}",
                    f"--nnodes={len(parsed_gpu_list)}",
                    f"--node_rank={index}",
                    f"--master_addr={parsed_gpu_list[0][0]}",
                    f"--master_port={master_port}",
                    script_path,
                ] + args
                torchrun_cmd = " ".join(torchrun_cmd)
                if prev_cmd is not None:
                    prev_cmds = " && ".join(prev_cmd)
                    remote_shell_cmd = f"{env_set_cmd} && {prev_cmds} && pwd && export CUDA_VISIBLE_DEVICES={','.join(gpus)} && {torchrun_cmd}"
                else:
                    remote_shell_cmd = f"{env_set_cmd} && pwd && export CUDA_VISIBLE_DEVICES={','.join(gpus)} && {torchrun_cmd}"
                ssh_cmd = f"ssh {ip} \"bash -c '{remote_shell_cmd}'\""
                process = subprocess.Popen(
                    ssh_cmd,
                    stdout=open(f"{WORK_DIR}/job_{job_id}/output.log", "a"),
                    stderr=open(f"{WORK_DIR}/job_{job_id}/error.log", "a"),
                    text=True,
                    shell=True,
                )
                self._process_list.append(process)
            # accept conn from rank0 and monitor the job
            server_instance = self._conn_job.accept()
            while True:
                cmd, data = server_instance.recv()
                if cmd == "init":
                    # send the job id to the client
                    server_instance.send("init", int(job_id))
                if cmd == "end":
                    # end the job
                    for process in self._process_list:
                        stdout, stderr = process.communicate()
                    self._conn_job.close()
                    server_instance.close()
                    self._conn_engine.send("end", "")
                    self._conn_engine.close()
                    return
                elif cmd == "alloc":
                    # try to allocate gpus
                    self._conn_engine.send("alloc", data)
                    cmd_response, new_gpu_list = self._conn_engine.recv()
                    assert cmd_response == "alloc"
                    server_instance.send("alloc", len(new_gpu_list))
                    if len(new_gpu_list) != len(gpu_list):  
                        # save checkpoint and restart
                        for process in self._process_list:
                            stdout, stderr = process.communicate()
                        server_instance.close()
                        gpu_list = new_gpu_list
                        break
                elif cmd == "GRs_target":
                    self._conn_engine.send("GRs_target", "")
                    cmd_response, GRs_target = self._conn_engine.recv()
                    assert cmd_response == "GRs_target"
                    server_instance.send("GRs_target", GRs_target)


@dataclass
class JobCard:
    _job_id: int
    _monitor_thread: threading.Thread
    _ideal_gpu_demand: float
    _occupy_gpu_num: int
    _status: Literal['running', 'pending', 'finished', 'failed']
    

class Engine:

    _allocator: Allocator
    _lock: threading.Lock
    _server: Server
    _server_port: int
    _now_job_id: int
    _job_cards: Dict[int, JobCard]
    _GRs_target: float
    _ideal_gpu_demand: float

    def __init__(self, server_port: int):
        self._allocator = Allocator()
        self._lock = threading.Lock()
        self._server = Server()
        self._server_port = server_port
        self._now_job_id = 0
        self._job_cards = {}
        self._GRs_target = GRS_TARGET
        self._ideal_gpu_demand = 0.0

    def add_device(self, node_ip: str, device_ids: List[int]):
        """
        Add GPUs in the node.
        """
        self._allocator.add_device(node_ip, device_ids)

    def run(self):
        """
        Run the engine and manage the jobs.
        """

        # start the server to listen to the job
        self._server.serve(get_ip(), self._server_port)
        maintain_GRs_target_thread = threading.Thread(target=self.matain_GRs_target, args=())
        maintain_GRs_target_thread.start()

        while True:
            conn_to_processor = self._server.accept()

            # initialize the job
            job_id = self.alloc_job_id()
            cmd, _ = conn_to_processor.recv()
            assert cmd == "init"
            conn_to_processor.send("init", job_id)

            # alloca required gpus for job
            cmd, data = conn_to_processor.recv()
            assert cmd == "alloc"
            gpu_list = self._allocator.alloca_job(job_id, self.get_adjusted_gpu_demand(data))
            if len(gpu_list) == 0:
                # no gpu available, put the job into pending
                job_pending_thread = threading.Thread(
                    target=self.pending_job, args=(job_id, conn_to_processor, data)
                )
                job_pending_thread.start()
            else:
                # get the resource, set the job as running
                job_monitor_thread = threading.Thread(
                    target=self.monitor_job, args=(job_id, conn_to_processor)
                )
                job_card = JobCard(job_id, job_monitor_thread, data, len(gpu_list), "running")
                with self._lock:
                    self._job_cards[job_id] = job_card
                    self._ideal_gpu_demand += data
                job_monitor_thread.start()
                conn_to_processor.send("alloc", gpu_list)

    def pending_job(self, job_id: int, conn_to_processor: ServerInstance, ideal_gpu_num: float):
        """
        Manage the pending job that is waiting for GPUs to be allocated.

        Args:
        job_id: The id of the job.
        conn_job: The connection to the job.
        gpu_num: The number of GPUs required by the job.
        """
        pending_times = 0
        job_monitor_thread = threading.Thread(
            target=self.monitor_job, args=(job_id, conn_to_processor),
        )
        with self._lock:
            job_card = JobCard(job_id, job_monitor_thread, ideal_gpu_num, 0, "pending")
            self._job_cards[job_id] = job_card
            self._ideal_gpu_demand += ideal_gpu_num
        while True:
            pending_times += 1
            time.sleep(PENDING_RETRY_INTERVAL)
            gpu_list = self._allocator.alloca_job(job_id, self.get_adjusted_gpu_demand(ideal_gpu_num))
            if len(gpu_list) > 0:
                # get the resource, remove the job from pending to running
                with self._lock:
                    self._job_cards[job_id]._occupy_gpu_num = len(gpu_list)
                    self._job_cards[job_id]._status = "running"
                job_monitor_thread.start()
                conn_to_processor.send("alloc", gpu_list)
                return
            elif pending_times <= MAX_PENDING_TIMES:
                # still pending, send the pending times to the job
                conn_to_processor.send("pending", pending_times)
            else:
                # pending times exceed the limit, close the connection
                print(
                    f"Fail to alloca resource for job_{job_id} after trying {MAX_PENDING_TIMES} times."
                )
                conn_to_processor.send("pending", pending_times)
                conn_to_processor.close()
                with self._lock:
                    self._job_cards[job_id]._status = "failed"
                    self._ideal_gpu_demand -= ideal_gpu_num
                return

    def monitor_job(self, job_id: int, conn_to_processor: ServerInstance):
        """
        Monitor the job and respond to the job's request to allocate or free GPUs, or end the job.
        """
        while True:
            cmd, data = conn_to_processor.recv()
            if cmd == "alloc":
                with self._lock:
                    self._ideal_gpu_demand += data - self._job_cards[job_id]._ideal_gpu_demand
                    self._job_cards[job_id]._ideal_gpu_demand = data
                result = self._allocator.alloca_job(job_id, self.get_adjusted_gpu_demand(data))
                with self._lock:
                    self._job_cards[job_id]._occupy_gpu_num = len(result)
                conn_to_processor.send("alloc", result)
            elif cmd == "end":
                self._allocator.free_job(job_id)
                conn_to_processor.close()
                with self._lock:
                    self._job_cards[job_id]._status = "finished"
                    self._job_cards[job_id]._occupy_gpu_num = 0
                    self._ideal_gpu_demand -= self._job_cards[job_id]._ideal_gpu_demand
                return
            elif cmd == "GRs_target":
                GRs_target = GRS_TARGET
                with self._lock:
                    GRs_target = self._GRs_target
                conn_to_processor.send("GRs_target", GRs_target)
                
    def matain_GRs_target(self):
        while True:
            with self._lock:
                ratio = self._allocator.total_gpu_num / self._ideal_gpu_demand
                if ratio < 1.0:
                    self._GRs_target = min(self._GRs_target + GRS_ADJUST_DELTA, GRS_UPPER_BOUND)
                else:
                    self._GRs_target = max(self._GRs_target - GRS_ADJUST_DELTA, GRS_LOWER_BOUND)
            time.sleep(MATAIN_GRS_TARGET_INTERVAL)
                

    def alloc_job_id(self) -> int:
        """
        Allocate a unique id for the job.

        Returns:
        int: A unique job id.
        """
        with self._lock:
            job_id = self._now_job_id
            self._now_job_id += 1
        return job_id
    
    def get_adjusted_gpu_demand(self, ideal_gpu_num: float) -> int:
        with self._lock:
            ratio = self._allocator.total_gpu_num / self._ideal_gpu_demand
        return math.ceil(ideal_gpu_num * min(1.0, ratio))

    



