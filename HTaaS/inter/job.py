import os
import subprocess
from typing import List

from .ipc import ClientIPC, Server, get_free_port, get_ip


class Job:
    def __init__(self):
        self._ipc = ClientIPC()
        self._server = Server()
        self._process_list: List[subprocess.Popen] = []

    def run(self, sched_ip: str, sched_port: int, script_path: str, args: List[str], add_cmd: List[str] = None):
        self._ipc.connect(sched_ip, sched_port)
        self._ipc.send('init', '')
        cmd, job_id = self._ipc.recv()
        print(f'JOB_ID:{job_id}...')
        os.makedirs(f'./tmp_{job_id}', exist_ok=True)
        assert cmd == 'init'
        self._ipc.send('alloc', 1)
        cmd, gpu_list = self._ipc.recv()
        if cmd == 'pending':
            while True:
                cmd, data = self._ipc.recv()
                if cmd == 'pending':
                    if int(data) > 20:
                        raise RuntimeError()
                elif cmd == 'alloc':
                    gpu_list = data
                    break
        self._server.serve(get_ip(), get_free_port())
        while True:
            parsed_list = {}
            for entry in gpu_list:
                ip, gpu_id = entry.split(':')
                if ip in parsed_list:
                    parsed_list[ip].append(gpu_id)
                else:
                    parsed_list[ip] = [gpu_id]
            parsed_gpu_list = [(ip, parsed_list[ip]) for ip, ids in parsed_list.items()]
            master_port = get_free_port(parsed_gpu_list[0][0], True)
            proxy_host = self._server.get_ip()
            proxy_port = self._server.get_port()
            cwd = script_path[:script_path.rfind('/')]
            env_set_cmd = [f"source {os.environ['CONDA_ACTIVATE_ADDR']} {os.environ['CONDA_ENV']}",
                           f"export PROXY_HOST={proxy_host}",
                           f"export PROXY_PORT={proxy_port}",
                           f"export PYTHONPATH={os.environ['PYTHONPATH']}",
                           f"cd {cwd}"
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
                print(f"bash -c '{remote_shell_cmd}'")
                process = subprocess.Popen(f"bash -c '{remote_shell_cmd}'", stdout=open(f'./tmp_{job_id}/output.log', 'a'), stderr=open(f'./tmp_{job_id}/error.log', 'a'), text=True, shell=True)

                ###### process = subprocess.Popen(ssh_cmd, stdout=open(f'./tmp_{job_id}/output.log', 'a'), stderr=open(f'./tmp_{job_id}/error.log', 'a'), text=True, shell=True)
                self._process_list.append(process)
            server_ipc = self._server.accept()  # accept conn from rank0
            while True:
                cmd, data = server_ipc.recv()
                if cmd == 'init':
                    server_ipc.send('init', int(job_id))
                if cmd == 'end':
                    for process in self._process_list:
                        stdout, stderr = process.communicate()
                    self._server.close()
                    server_ipc.close()
                    self._ipc.send('end', '')
                    self._ipc.close()
                    return
                elif cmd == 'alloc':
                    gpu_num = int(data)
                    self._ipc.send('alloc', gpu_num)
                    cmd_response, new_gpu_list = self._ipc.recv()
                    assert cmd_response == 'alloc'
                    assert len(new_gpu_list) >= len(gpu_list)
                    server_ipc.send('alloc', len(new_gpu_list))
                    if len(new_gpu_list) > len(gpu_list):  # save checkpoint and restart on more gpus
                        for process in self._process_list:
                            stdout, stderr = process.communicate()
                        server_ipc.close()
                        gpu_list = new_gpu_list
                        break
                elif cmd == 'free':
                    self._ipc.send('free', int(data))
                    cmd_response, new_gpu_list = self._ipc.recv()
                    assert cmd_response == 'free'
                    assert len(new_gpu_list) == int(data)
                    for process in self._process_list:  # save checkpoint and restart on less gpus
                        stdout, stderr = process.communicate()
                    server_ipc.close()
                    gpu_list = new_gpu_list
                    break
                elif cmd == 'status':
                    self._ipc.send('status', '')
                    cmd_response, standard = self._ipc.recv()
                    assert cmd_response == 'status'
                    server_ipc.send('status', (float(standard)))
                elif cmd == 'heartbeat':
                    self._ipc.send('heartbeat', int(data))
