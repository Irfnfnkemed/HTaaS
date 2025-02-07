import os
from HTaaS.inter.job import Job

if __name__ == '__main__':
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".")) + '/main.py'
    job = Job()
    job.run('192.168.1.72', 12345, script_path, [])
