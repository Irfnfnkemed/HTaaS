from FTaaS.inter.scheduler import Scheduler
from FTaaS.inter.ipc import get_ip

if __name__ == '__main__':
    print(get_ip())
    scheduler = Scheduler(12345)
    scheduler.set_device('10.14.5.3', [0,1,2,3])
    scheduler.run()
