# import pandas as pd
# df  = pd.read_csv('network_latency-old.csv')

# from sklearn.linear_model import LinearRegression

# X = df[['input_length']]
# y = df['latency']

# model = LinearRegression().fit(X, y)

# print(model.coef_)
# print('bandwidth = ', 1 / model.coef_[0], 'token/s')
# print('overhead = ', model.intercept_, 's')

# # print(model.predict(X))

# print(model.score(X, y))

# import matplotlib.pyplot as plt
# plt.scatter(X, y)
# plt.plot(X, model.predict(X), color='red')
# plt.savefig('network_latency_model.png')

import json 

with open('profile_events.jsonl', 'r') as f:
    events = json.load(f)

cared_events = [event for event in events if event['event_type'] in ['arrival', 'kv_xfer_ready', 'finish']]
data = []
for i in range(len(cared_events) - 1):
    if cared_events[i]['event_type'] == 'kv_xfer_ready':
        j = i - 1
        while j >= 0 and cared_events[j]['event_type'] != 'finish' and cared_events[j]['request_id'] != cared_events[i]['request_id']:
            j -= 1
        assert j >= 0
        
        kv_xfer_event = cared_events[i]
        while j >= 0 and cared_events[j]['event_type'] != 'arrival' and cared_events[j]['request_id'] != cared_events[i]['request_id']:
            j -= 1
        assert j >= 0
        arrival_event = cared_events[j]
        finish_event = cared_events[j]
        data.append((kv_xfer_event['timestamp'] - finish_event['timestamp'], arrival_event['prompt_tokens']))

data = data[len(data) // 2:]
data = [x for x in data if x[1] < 10000]

print(data)
import matplotlib.pyplot as plt
X = [[x[1]] for x in data]
y = [x[0] for x in data]
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)

print(model.coef_)
print('bandwidth = ', 1 / model.coef_[0], 'token/s')
print('overhead = ', model.intercept_, 's')

# print(model.predict(X))

print(model.score(X, y))

import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.savefig('network_latency_model.png')

plt.scatter(X, y)
plt.savefig('network_latency_model.png')