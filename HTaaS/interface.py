from typing import List, Optional
from .engine import Engine, Job
from .comm import *


class EngineInterface:
    
    _inner_engine: Engine
    
    def __init__(self, port: int = 0):
        if port == 0:
            port = get_free_port()
        self._inner_engine = Engine(port)
        
    def run(self):
        self._inner_engine.run()
        
    def add_devices(self, node_ip: str, devices_ids: List[int]):
        self._inner_engine.add_device(node_ip, devices_ids)
        

class UserInterface:
    
    def create_job(self, engine_ip: str, engine_port: int, job_script_path: str, 
                   job_args: List[str], prev_cmd: Optional[List[str]]):
        job = Job()
        job.run(engine_ip, engine_port, job_script_path, job_args, prev_cmd)
        
    