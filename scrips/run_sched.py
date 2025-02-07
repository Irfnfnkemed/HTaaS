from HTaaS.inter.scheduler import Scheduler
from HTaaS.inter.ipc import get_ip

if __name__ == '__main__':
    print(get_ip())
    scheduler = Scheduler(12345)
    scheduler.set_device('192.168.1.72', [4,5,6,7])
    scheduler.run()
