from dataclasses import dataclass
import os
import threading
import time
from typing import List, Dict

from .manager import Allocator
from .comm import *
from .config import *

"""
This module provides the JobCard class, which is responsible for managing and monitoring the single job process.
"""


@dataclass
class JobCard:
    job_id: int
    monitor: threading.Thread
    gpu_num: int = 0
    conn_engine: ClientInstance = ClientInstance()
    conn_job: Server = Server()
    process_list: List[subprocess.Popen] = []

    def run(
        self,
        engine_ip: str,
        engine_port: int,
        script_path: str,
        args: List[str],
        add_cmd: List[str] = None,
    ):
        """
        Launches, run and monitor the job.
        """

        # connect to the engine and prepare the job
        self.conn_engine.connect(engine_ip, engine_port)
        self.conn_engine.send("init", "")
        cmd, job_id = self.conn_engine.recv()
        print(f"JOB_ID:{job_id}...")
        os.makedirs(f"{WORK_DIR}/job_{job_id}", exist_ok=True)
        assert cmd == "init"
        # alloca 1 gpu to profile the job
        self.conn_engine.send("alloc", 1)
        cmd, gpu_list = self.conn_engine.recv()
        if cmd == "pending":
            while True:
                cmd, data = self.conn_engine.recv()
                if cmd == "pending":
                    if int(data) > MAX_PENDING_TIMES:
                        raise RuntimeError()
                elif cmd == "alloc":
                    gpu_list = data
                    break
        self.conn_job.serve(get_ip(), get_free_port())
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
            self.process_list.clear()
            master_port = get_free_port(parsed_gpu_list[0][0], True)
            job_host = self.conn_job.get_ip()
            job_port = self.conn_job.get_port()
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
                if add_cmd is not None:
                    additional = " && ".join(add_cmd)
                    remote_shell_cmd = f"{env_set_cmd} && {additional} && pwd && export CUDA_VISIBLE_DEVICES={','.join(gpus)} && {torchrun_cmd}"
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
                self.process_list.append(process)
            # accept conn from rank0 and monitor the job
            server_instance = self.conn_job.accept()
            while True:
                cmd, data = server_instance.recv()
                if cmd == "init":
                    # send the job id to the client
                    server_instance.send("init", int(job_id))
                if cmd == "end":
                    # end the job
                    for process in self.process_list:
                        stdout, stderr = process.communicate()
                    self.conn_job.close()
                    server_instance.close()
                    self.conn_engine.send("end", "")
                    self.conn_engine.close()
                    return
                elif cmd == "alloc":
                    # try to allocate gpus
                    gpu_num = int(data)
                    self.conn_engine.send("alloc", gpu_num)
                    cmd_response, new_gpu_list = self.conn_engine.recv()
                    assert cmd_response == "alloc"
                    server_instance.send("alloc", len(new_gpu_list))
                    if len(new_gpu_list) != len(
                        gpu_list
                    ):  # save checkpoint and restart
                        for process in self.process_list:
                            stdout, stderr = process.communicate()
                        server_instance.close()
                        gpu_list = new_gpu_list
                        break
