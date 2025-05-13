import threading
import time
from typing import Dict, List
from .manager import Allocator
from .processor import JobCard
from .comm import *
from .config import *

"""
This module provides the Engine class, which schedules all the jobs.
"""


class Engine:

    allocator: Allocator
    lock: threading.Lock
    server: Server
    server_port: int
    now_job_id: int
    job_cards: Dict[int, JobCard]
    pending_jobs: Dict[int, int]  # job_id -> pending_times

    def __init__(self, server_port: int):
        self.allocator = Allocator()
        self.lock = threading.Lock()
        self.server = Server()
        self.server_port = server_port
        self.now_job_id = 0
        self.job_cards = {}
        self.pending_jobs: Dict[int, int] = {}

    def set_device(self, node_ip: str, device_ids: List[int]):
        """
        Set GPUs for the node.
        """
        self.allocator.set_device(node_ip, device_ids)

    def run(self):
        """
        Run the engine and manage the jobs.
        """

        # start the server to listen to the job
        self.server.serve(get_ip(), self.server_port)

        while True:
            conn_job = self.server.accept()

            # initialize the job
            job_id = self.alloc_job_id()
            cmd, _ = conn_job.recv()
            assert cmd == "init"
            conn_job.send("init", job_id)

            # alloca required gpus for job
            cmd, data = conn_job.recv()
            assert cmd == "alloc"
            gpu_list = self.allocator.alloca(job_id, int(data))
            if len(gpu_list) == 0:
                # no gpu available, put the job into pending
                job_pending_thread = threading.Thread(
                    target=self.pending_job, args=(job_id, conn_job, int(data))
                )
                job_pending_thread.start()
            else:
                # get the resource, set the job as running
                conn_job.send("alloc", gpu_list)
                job_monitor_thread = threading.Thread(
                    target=self.monitor_job,
                    args=(
                        job_id,
                        conn_job,
                    ),
                )
                job_card = JobCard(job_id, job_monitor_thread)
                with self.lock:
                    self.job_cards[job_id] = job_card
                job_monitor_thread.start()

    def pending_job(self, job_id: int, conn_job: ServerInstance, gpu_num: int):
        """
        Manage the pending job that is waiting for GPUs to be allocated.

        Args:
        job_id: The id of the job.
        conn_job: The connection to the job.
        gpu_num: The number of GPUs required by the job.
        """
        cnt = 0
        with self.lock:
            self.pending_jobs[job_id] = 0
        while True:
            cnt += 1
            with self.lock:
                self.pending_jobs[job_id] = cnt
            time.sleep(PENDING_RETRY_INTERVAL)
            gpu_list = self.allocator.alloca(job_id, int(gpu_num))
            if len(gpu_list) > 0:
                # get the resource, remove the job from pending to running
                conn_job.send("alloc", gpu_list)
                job_monitor_thread = threading.Thread(
                    target=self.monitor_job,
                    args=(
                        job_id,
                        conn_job,
                    ),
                )
                job_card = JobCard(job_id, job_monitor_thread)
                with self.lock:
                    self.job_cards[job_id] = job_card
                    self.pending_jobs.pop(job_id)
                job_monitor_thread.start()
                return
            elif cnt <= MAX_PENDING_TIMES:
                # still pending, send the pending times to the job
                conn_job.send("pending", cnt)
            else:
                # pending times exceed the limit, close the connection
                print(
                    f"Fail to alloca resource for job_{job_id} after trying {MAX_PENDING_TIMES} times."
                )
                conn_job.send("pending", cnt)
                conn_job.close()
                with self.lock:
                    self.pending_jobs.pop(job_id)
                return

    def monitor_job(self, job_id: int, conn_job: ServerInstance):
        """
        Monitor the job and respond to the job's request to allocate or free GPUs, or end the job.
        """
        while True:
            cmd, data = conn_job.recv()
            if cmd == "alloc":
                result = self.allocator.alloca(job_id, int(data))
                with self.lock:
                    self.job_cards[job_id].gpu_num = len(result)
                conn_job.send("alloc", result)
            elif cmd == "free":
                result = self.allocator.alloca(job_id, int(data))
                assert len(result) == int(data)
                with self.lock:
                    self.job_cards[job_id].gpu_num = int(data)
                conn_job.send("free", result)
            elif cmd == "end":
                self.allocator.free(job_id)
                conn_job.close()
                with self.lock:
                    self.job_cards.pop(job_id)
                return

    def alloc_job_id(self) -> int:
        """
        Allocate a unique id for the job.

        Returns:
        int: A unique job id.
        """
        with self.lock:
            job_id = self.now_job_id
            self.now_job_id += 1
        return job_id
