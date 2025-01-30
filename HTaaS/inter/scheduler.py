import threading
import time
from typing import List, Dict

from .alloc import Allocator
from .ipc import ServerIPC, Server, get_ip


class JobCard:
    def __init__(self, job_id: int, monitor: threading.Thread):
        self.job_id = job_id
        self.monitor = monitor
        self.gpu_num = 0
        self.scarcity = 0
        self.adjust_standard = False


class Scheduler:
    def __init__(self, port: int):
        self._allocator = Allocator()
        self._job_cards: Dict[int, JobCard] = {}
        self._cnt = 0
        self._lock = threading.Lock()
        self._server = Server()
        self._server_port = port

        self._epb_standard = 0.0
        self._epb_standard_upper = 1.0
        self._epb_standard_lower = -1.0
        self._epb_standard_adjust_step = 0.1

        self._pending_jobs: Dict[int, int] = {}

        self._cluster_status = 0

        self._adjust_epb_standard = None
        self._monitor_cluster_status = None

    def set_device(self, node_ip: str, device_id: List[int]):
        self._allocator.set_device(node_ip, device_id)

    def run(self):

        self._server.serve(get_ip(), self._server_port)
        self._adjust_epb_standard = threading.Thread(target=self.adjust_epb_standard, args=())
        self._monitor_cluster_status = threading.Thread(target=self.monitor_cluster_status, args=())
        self._adjust_epb_standard.start()
        self._monitor_cluster_status.start()

        while True:
            job_ipc = self._server.accept()
            job_id = self.alloc_id()
            cmd, _ = job_ipc.recv()
            assert cmd == 'init'
            job_ipc.send('init', job_id)
            cmd, data = job_ipc.recv()
            assert cmd == 'alloc'
            gpu_list = self._allocator.alloca(job_id, int(data))
            if len(gpu_list) == 0:
                job_pending_thread = threading.Thread(target=self.pending_job, args=(job_id, job_ipc, int(data)))
                job_pending_thread.start()
            else:
                job_ipc.send('alloc', gpu_list)
                job_monitor_thread = threading.Thread(target=self.monitor_job, args=(job_id, job_ipc,))
                job_card = JobCard(job_id, job_monitor_thread)
                with self._lock:
                    self._job_cards[job_id] = job_card
                job_monitor_thread.start()

    def pending_job(self, job_id: int, job_ipc: ServerIPC, num: int):
        cnt = 0
        with self._lock:
            self._pending_jobs[job_id] = 0
        while True:
            cnt += 1
            with self._lock:
                self._pending_jobs[job_id] = cnt
            time.sleep(10)
            gpu_list = self._allocator.alloca(job_id, int(num))
            if len(gpu_list) > 0:
                job_ipc.send('alloc', gpu_list)
                job_monitor_thread = threading.Thread(target=self.monitor_job, args=(job_id, job_ipc,))
                job_card = JobCard(job_id, job_monitor_thread)
                with self._lock:
                    self._job_cards[job_id] = job_card
                    self._pending_jobs.pop(job_id)
                job_monitor_thread.start()
                return
            elif cnt <= 100:
                job_ipc.send('pending', cnt)
            else:
                print(f'Fail to alloca resource for job{job_id}.')
                job_ipc.send('pending', cnt)
                job_ipc.close()
                with self._lock:
                    self._pending_jobs.pop(job_id)
                return

    def monitor_job(self, job_id: int, job_ipc: ServerIPC):
        while True:
            cmd, data = job_ipc.recv()
            if cmd == 'alloc':
                result = self._allocator.alloca(job_id, int(data))
                with self._lock:
                    if len(result) < int(data):  # Lack of resources
                        self._job_cards[job_id].scarcity += 1
                    else:
                        self._job_cards[job_id].scarcity = 0
                    self._job_cards[job_id].gpu_num = len(result)
                job_ipc.send('alloc', result)
            elif cmd == 'free':
                result = self._allocator.alloca(job_id, int(data))
                assert len(result) == int(data)
                with self._lock:
                    self._job_cards[job_id].scarcity = 0
                    self._job_cards[job_id].gpu_num = int(data)
                job_ipc.send('free', result)
            elif cmd == 'end':
                self._allocator.free(job_id)
                job_ipc.close()
                with self._lock:
                    self._job_cards.pop(job_id)
                return
            elif cmd == 'status':
                with self._lock:
                    standard = self._epb_standard
                    self._job_cards[job_id].adjust_standard = True
                job_ipc.send('status', standard)
            elif cmd == 'heartbeat':
                with self._lock:
                    self._job_cards[job_id].scarcity += int(data)

    def alloc_id(self) -> int:
        with self._lock:
            job_id = self._cnt
            self._cnt += 1
            return job_id

    def monitor_cluster_status(self):
        while True:
            with self._lock:
                # Scan running jobs
                scarcity = 0
                for job_card in self._job_cards.values():
                    if job_card.scarcity >= 1:
                        scarcity += 1

                if scarcity > len(self._job_cards) / 2:
                    self._cluster_status = 1  # Busy
                else:
                    gpus_num, gpus_busy_num = self._allocator.get_status()
                    if gpus_num > gpus_busy_num:
                        self._cluster_status = -1  # Too idle
                    else:
                        self._cluster_status = 0  # Ideal

                # Scan pending jobs
                for pending_times in self._pending_jobs.values():
                    if pending_times >= 5:
                        self._cluster_status = max(self._cluster_status, 1)

            time.sleep(5)

    def adjust_epb_standard(self):
        while True:
            with self._lock:
                # Whether epb-standard adjustment has been responded
                already_adjust = 0
                for job_card in self._job_cards.values():
                    if job_card.adjust_standard:
                        already_adjust += 1

                if already_adjust > len(self._job_cards) / 2:
                    if self._cluster_status == -1:
                        self._epb_standard = max(self._epb_standard_lower, self._epb_standard - self._epb_standard_adjust_step)
                        for job_card in self._job_cards.values():
                            job_card.adjust_standard = False
                    elif self._cluster_status == 1:
                        self._epb_standard = min(self._epb_standard_upper, self._epb_standard + self._epb_standard_adjust_step)
                        for job_card in self._job_cards.values():
                            job_card.adjust_standard = False

            time.sleep(10)
