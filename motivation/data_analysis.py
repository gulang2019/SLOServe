import pandas as pd
import numpy as np
from Dataset.dataset import ArrivalTimes, Requests


for arrival_times_name in [
    # 'burstgpt_GPT-4_API log', 
    #                        'burstgpt_GPT-4_Conversation log',
    #                        'burstgpt_ChatGPT_API log',
    #                        'burstgpt_ChatGPT_Conversation log',
                           'azure_code', 'azure_chat']:
    data = ArrivalTimes.load(arrival_times_name)

    data.visualize()

# for dataset in ['arxiv_summary', 'humaneval',
#                 'deepseek-r1', 'azure_code',
#                 'azure_chat', 'sharegpt_chat']:
#     data = Requests.load(dataset)
#     data.visualize()
