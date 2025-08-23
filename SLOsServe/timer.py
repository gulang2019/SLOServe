import time 

class Timer:
    def __init__(self, name):
        self.name = name
        self.starts = {}
        self.times = {}

    def start(self, name):
        self.starts[name] = time.perf_counter()
    
    def stop(self, name):
        self.times[name] = self.times.get(name, 0) + time.perf_counter() - self.starts.pop(name)

    def report(self):
        times = sorted(list(self.times.items()), key = lambda x: x[1], reverse=True)
        tot_time = sum(self.times.values())
        print(f'Timer: {self.name}')
        for name, time in times:
            print(f'{name}\t{round(time, 2)}\t{round(time / tot_time * 100, 2)}%s')

    def reset(self):
        self.starts = {}
        self.times = {}