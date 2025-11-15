filename = 'engine_core_0.events.jsonl'

import json

with open(filename, 'r') as f:
    events = json.load(f)

req_id_2_id = {}

for event in events:
    if 'request_id' in event:
        if event['request_id'] not in req_id_2_id:
            req_id_2_id[event['request_id']] = str(len(req_id_2_id))

for event in events:
    if 'request_id' in event:
        event['request_id'] = req_id_2_id[event['request_id']]
    if event['event_type'] == 'schedule_problem':
        for req in event['reqs']:
            req['id'] = req_id_2_id[req['id']]
        event['accepted_ids'] = [req_id_2_id[req_id] for req_id in event['accepted_ids']]
        for batch in event['batch_schedule']:
            batch['id'] = req_id_2_id[batch['id']]
    if event['event_type'] == 'batch':
        event['req_ids'] = [req_id_2_id[req_id] for req_id in event['req_ids']]
        event['num_scheduled_tokens'] = {req_id_2_id[req_id]: num_scheduled_tokens for req_id, num_scheduled_tokens in event['num_scheduled_tokens'].items()}


with open('engine_core_0_analyzed.events.jsonl', 'w') as f:
    json.dump(events, f, indent=4)