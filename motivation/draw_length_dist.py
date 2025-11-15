from Dataset.dataset import Requests


for requests_name in ['deepseek-r1', 'azure_code', 'azure_chat']:
    requests = Requests.load(requests_name, 16384)
    # draw the distribution of input_length, output_length, thinking_length 
    requests.visualize(fit_with='lognorm')
    requests.visualize(log_scale=False, fit_with='lognorm')

requests = Requests.merge(Requests.load('azure_chat', 16384), 
                          Requests.load('azure_code', 16384))
requests.visualize()
requests.visualize(log_scale=False)