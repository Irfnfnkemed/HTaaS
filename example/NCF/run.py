import os
from FTaaS.inter.job import Job

if __name__ == '__main__':
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".")) + '/main.py'
    job = Job()
    job.run('10.14.5.8', 12345, script_path, [])
