import subprocess
import os
import pandas as pd
import time
import json

class Launcher:
    def __init__(self, trace_file, info_file):
        df = pd.read_csv(trace_file)
        self.trace = df.to_dict(orient='list')
        with open(info_file, 'r') as file:
            self.info = json.load(file)
        self.now_index = 0
        self.beg_time = 0
        self.process_list = []
        
    
    def run(self):
        self.beg_time = time.time()
        while self.now_index < len(self.trace['time']):
            if time.time() - self.beg_time >= self.trace['time'][self.now_index]:
                self.launch()
                self.now_index += 1
            time.sleep(1)
        self.wait()

    def launch(self):
        if self.trace['application'][self.now_index] in self.info['application']:
            print(f"Launch {self.trace['name'][self.now_index]} at time {time.time() - self.beg_time}")
            cmd = f"python3 {self.info['info'][self.trace['application'][self.now_index]]['path']}"
            print(cmd)
            process = subprocess.Popen(cmd, stdout=open(f"./{self.trace['name'][self.now_index]}_output.log", 'a'), stderr=open(f"./{self.trace['name'][self.now_index]}_error.log", 'a'), text=True, shell=True)
            self.process_list.append(process)
        else:
            print(f"Unkown application {self.trace['name'][self.now_index]} at time {time.time() - self.beg_time}")

    def wait(self):
        for process in self.process_list:
            process.wait()

if __name__ == "__main__":
    launcher = Launcher('trace.csv', 'accept.json')
    launcher.run()
        
