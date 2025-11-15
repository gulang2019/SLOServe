from Dataset.dataset import ArrivalTimes, Requests, Request
import matplotlib.pyplot as plt
import os 

def main(
    requests_name: str,
    arrival_times_name: str,
    slo_ttft: float,
    slo_tpot: float,
):
    requests = Requests.load(requests_name)
    arrival_times = ArrivalTimes.load(arrival_times_name)

    class RequestInstance:
        arrival_time: float
        request: Request
        
        def __init__(self, arrival_time: float, request: Request):
            self.arrival_time = arrival_time
            self.request = request
        
    request_instances = []
    for req, arrival_time in zip(requests.requests, arrival_times.arrival_times):
        request_instances.append(RequestInstance(arrival_time, req))

    req_instances = sorted(request_instances, key=lambda x: x.arrival_time)
    
    # rolling_loads = []
    loads_event = []
    for req_instance in req_instances:
        req_instance.request.input_length
        loads_event.append((req_instance.arrival_time, 
                            req_instance.request.input_length / slo_ttft))
        loads_event.append((
            req_instance.arrival_time + slo_ttft,
            -req_instance.request.input_length / slo_ttft
        ))
        loads_event.append((
            req_instance.arrival_time + slo_ttft,
            1 / slo_tpot
        ))
        loads_event.append((
            req_instance.arrival_time + slo_ttft + slo_tpot * req_instance.request.output_length,
            -1 / slo_tpot
        ))
        
    loads_event = sorted(loads_event, key=lambda x: x[0])
    
    rolling_loads = [(0,0)]
    for event in loads_event:
        rolling_loads.append((event[0] - 1e-6, rolling_loads[-1][1]))
        rolling_loads.append((event[0], rolling_loads[-1][1] + event[1]))
        
    # Make the figure wider
    fig, ax = plt.subplots(figsize=(16, 5))  # Increased width from 10 to 16
    times, loads = zip(*rolling_loads)

    # Add smoothing to the load curve using a moving average
    import numpy as np
    window_size = 20  # You can adjust this for more/less smoothing
    loads_np = np.array(loads)
    pad = np.ones(window_size-1) * loads_np[0]
    padded_loads = np.concatenate([pad, loads_np])
    smoothed_loads = np.convolve(padded_loads, np.ones(window_size)/window_size, mode='valid')
    ax.plot(times, loads)
    ax.set_xlabel('Time')
    ax.set_ylabel('Load')
    ax.set_title('Token Budget Requirement')
    fig.savefig(f'figs/token_budget_requirement/{requests_name}_{arrival_times_name}_{slo_ttft}_{slo_tpot}.png')
    print(f'Saved figs/token_budget_requirement/{requests_name}_{arrival_times_name}_{slo_ttft}_{slo_tpot}.png')
    
if __name__ == '__main__':
    os.makedirs('figs/token_budget_requirement', exist_ok=True)
    for requests_name in ['azure_chat_23', 'azure_code_23']:
        for arrival_times_name in ['azure_chat_23', 'azure_code_23']:
            for slo_ttft in [0.5]:
                for slo_tpot in [0.1]:
                    main(requests_name, arrival_times_name, slo_ttft, slo_tpot)
        
        