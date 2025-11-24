from dataclasses import dataclass, field
from typing import List, Optional
import pickle
import os
import datasets
from transformers import AutoTokenizer
import random

from Dataset.dataset import Request, Requests, ArrivalTimes

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1')

DATASET_DIR = 'assets/datasets'

os.makedirs(DATASET_DIR, exist_ok=True)

def download_dataset(dataset_name: str): 
    logger.info(f"Starting download for dataset: {dataset_name}")

    def count_length(text: str): 
        if text is None:
            return 0
        length = len(tokenizer.encode(text))
        logger.debug(f"Counted {length} tokens for text: {text[:50]}..." if text else "None")
        return length
    
    requests = []
    arrival_times = None

    if dataset_name == 'deepseek-r1':
        logger.info("Loading 'open-r1/Mixture-of-Thoughts' dataset (train split)...")
        dataset = datasets.load_dataset('open-r1/Mixture-of-Thoughts', 'all')
        dataset = dataset['train']
        logger.info(f"Loaded dataset with {len(dataset)} examples.")
        # dataset = dataset.select(range(5000))
        def process(example): 
            messages = example['messages']
            prompt = messages[0]['content']
            response = messages[1]['content']
                        
            # Split the response into thinking and answer parts.
            # The thinking part is wrapped in <think></think>
            import re
            # Find the <think>...</think> block
            thinking_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            if thinking_match:
                thinking = thinking_match.group(1).strip()
                # The answer is everything after </think>
                answer_start = thinking_match.end()
                answer = response[answer_start:].strip()
                logger.debug(f"Extracted thinking and answer from response. Thinking: {thinking[:30]}..., Answer: {answer[:30]}...")
            else:
                # If no <think> tag found, treat all as answer, thinking is empty
                thinking = ""
                answer = response.strip()
                logger.debug("No <think> tag found. Entire response treated as answer.")
            return {
                "prompt": prompt,
                "thinking": thinking,
                "answer": answer,
                "input_length": count_length(prompt),
                "thinking_length": count_length(thinking),
                "output_length": count_length(answer)
            }
        logger.info("Processing dataset examples...")
        processed = dataset.map(process)
        logger.info(f"Processed {len(processed)} examples.")
        
        # Process requests in parallel
        import concurrent.futures
        def create_request(ex):
            return Request(
                prompt=ex['prompt'],
                thinking=ex['thinking'],
                answer=ex['answer'],
                input_length=ex['input_length'],
                thinking_length=ex['thinking_length'],
                output_length=ex['output_length']
            )
        
        logger.info("Creating Request objects in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            request_futures = [executor.submit(create_request, ex) for ex in processed]
            
            for idx, future in enumerate(concurrent.futures.as_completed(request_futures)):
                if idx % 1000 == 0 and idx > 0:
                    logger.info(f"Processed {idx} requests so far...")
                requests.append(future.result())
        logger.info(f"Finished processing all requests for {dataset_name}.")
    elif dataset_name in ['azure_code', 'azure_chat', 'azure_chat_23', 'azure_code_23']:
        url = {
            'azure_code': 'https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_code_1week.csv',
            'azure_chat': 'https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_conv_1week.csv',
            'azure_chat_23': 'https://raw.githubusercontent.com/Azure/AzurePublicDataset/refs/heads/master/data/AzureLLMInferenceTrace_conv.csv',
            'azure_code_23': 'https://raw.githubusercontent.com/Azure/AzurePublicDataset/refs/heads/master/data/AzureLLMInferenceTrace_code.csv'
        }
        logger.info(f"Downloading CSV from {url[dataset_name]}")
        '''
        Schema
        Field	Description
        TIMESTAMP	Invocation time example: 2024-05-10 00:00:00.009930+00:00
        ContextTokens (int)	Number of context tokens
        GeneratedTokens (int)	Number of generated tokens
        '''
        import pandas as pd
        from datetime import datetime
        df = pd.read_csv(url[dataset_name])
        df = df.iloc[:100000]
        logger.info(f"Loaded CSV with {len(df)} rows.")
        # Robustly parse timestamps with or without microseconds
        def parse_timestamp(x):
            # the 23 trace has format 2023-11-16 18:17:03.9799600
            try:
                if dataset_name == 'azure_chat_23' or dataset_name == 'azure_code_23':
                    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f0')
                return datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f%z')
            except ValueError:
                return datetime.strptime(x, '%Y-%m-%d %H:%M:%S%z')
        arrival_times = df['TIMESTAMP'].apply(parse_timestamp)
        # make arrival times a list of floats starting from 0
        arrival_times = arrival_times.apply(lambda x: (x - arrival_times.iloc[0]).total_seconds())
        logger.info("Parsed arrival times and normalized to seconds since first request.")
        
        # Process requests in parallel
        import concurrent.futures
        def create_azure_request(row):
            return Request(
                input_length=int(row['ContextTokens']),
                output_length=int(row['GeneratedTokens'])
            )
        
        logger.info("Creating Request objects in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            request_futures = [executor.submit(create_azure_request, row) for _, row in df.iterrows()]
            
            for idx, future in enumerate(concurrent.futures.as_completed(request_futures)):
                if idx % 10000 == 0 and idx > 0:
                    logger.info(f"Processed {idx} requests so far...")
                requests.append(future.result())
        arrival_times = arrival_times.tolist()
        logger.info(f"Total requests processed: {len(requests)}")
        arrival_times_obj = ArrivalTimes(dataset_name, arrival_times)
        logger.info(f"Saving arrival times for {dataset_name}...")
        arrival_times_obj.save()
        logger.info("Arrival times saved.")
    
    elif dataset_name == 'sharegpt_chat':
        dataset_map = {
            'sharegpt_chat': 'LNTANOooo/sharegpt52k',
        }
        dataset_id = dataset_map[dataset_name]
        dataset = datasets.load_dataset(dataset_id)
        dataset = dataset['train']
        dataset = dataset.select(range(3000))
        logger.info(f"Loaded dataset with {len(dataset)} examples.")
        def format_conversation(conversation):
            return '\n'.join([f'{item.get("role", "")}: {item.get("content", "")}' for item in conversation])

        # Collect all valid conversations first
        valid_conversations = []
        for idx, ex in enumerate(dataset):
            for i, item in enumerate(ex['conversation']):
                if item['role'] == 'assistant':
                    previous_items = ex['conversation'][:i]
                    # if i - 2 >= 0:
                    #     cached_contents = format_conversation(ex['conversation'][:i-2])
                    # else: 
                    #     cached_contents = ""
                    # cached_contents = format_conversation(previous_items)
                    prompt = format_conversation(previous_items)
                    response = item.get("content", "")
                    valid_conversations.append(("", prompt, response))

        # Process conversations in parallel
        import concurrent.futures
        def create_sharegpt_request(prompt_response):
            cached_contents, prompt, response = prompt_response
            return Request(
                prompt=prompt,
                answer=response,
                cached_length = 0,
                input_length=count_length(prompt),
                output_length=count_length(response)
            )

        logger.info("Creating Request objects in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            request_futures = [executor.submit(create_sharegpt_request, conv) for conv in valid_conversations]

            for idx, future in enumerate(concurrent.futures.as_completed(request_futures)):
                if idx % 1000 == 0 and idx > 0:
                    logger.info(f"Processed {idx} requests so far...")
                requests.append(future.result())
            random.shuffle(requests)
        logger.info(f"Finished processing all requests for {dataset_name}.")
    elif dataset_name == 'sharegpt_code':
        dataset_id = 'QuixiAI/Code-290k-ShareGPT-Vicuna'
        dataset = datasets.load_dataset(dataset_id)
        dataset = dataset['train']
        dataset = dataset.select(range(5000))
        logger.info(f"Loaded dataset with {len(dataset)} examples.")
        def format_conversation(conversation):
            return '\n'.join([f'{item.get("from", "")}: {item.get("value", "")}' for item in conversation])

        # Collect all valid conversations first
        valid_conversations = []
        for idx, ex in enumerate(dataset):
            for i, item in enumerate(ex['conversations']):
                if item['from'] == 'gpt':
                    previous_items = ex['conversations'][:i]
                    # if i - 2 >= 0:
                    #     cached_contents = format_conversation(ex['conversations'][:i-2])
                    # else: 
                    #     cached_contents = ""
                    prompt = ex['conversations'][i-1].get("value", "") # format_conversation(ex['conversations'][i-1])
                    response = item.get("value", "")
                    valid_conversations.append(("", prompt, response))

        # Process conversations in parallel
        import concurrent.futures
        def create_sharegpt_request(prompt_response):
            cached_contents, prompt, response = prompt_response
            return Request(
                prompt=prompt,
                answer=response,
                cached_length=0,
                input_length=count_length(prompt),
                output_length=count_length(response)
            )

        logger.info("Creating Request objects in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            request_futures = [executor.submit(create_sharegpt_request, conv) for conv in valid_conversations]

            for idx, future in enumerate(concurrent.futures.as_completed(request_futures)):
                if idx % 1000 == 0 and idx > 0:
                    logger.info(f"Processed {idx} requests so far...")
                requests.append(future.result())
        logger.info(f"Finished processing all requests for {dataset_name}.")
    elif dataset_name == 'burstgpt':
        import pandas as pd
        df = pd.read_csv('Dataset/BurstGPT_without_fails_1.csv')
        
        logger.info(f"Loaded CSV with {len(df)} rows.")


        for (model, log_type), tdf in df.groupby(['Model', 'Log Type']):
            # Process requests in parallel
            import concurrent.futures
            def create_burstgpt_request(row):
                return Request(
                    input_length=row['Request tokens'],
                    output_length=row['Response tokens'],
                ), row['Timestamp']
            requests = []
            logger.info("Creating Request objects in parallel...")
            for _, row in tdf.iterrows():
                requests.append(create_burstgpt_request(row))
            # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            #     request_futures = [executor.submit(create_burstgpt_request, row) for _, row in tdf.iterrows()]
                
            #     for idx, future in enumerate(concurrent.futures.as_completed(request_futures)):
            #         if idx % 1000 == 0 and idx > 0:
            #             logger.info(f"Processed {idx} requests so far...")
            #         requests.append(future.result())
            requests, arrival_times = zip(*requests)
            requests_obj = Requests(f'{dataset_name}_{model}_{log_type}', requests)
            requests_obj.save()
            arrival_times_obj = ArrivalTimes(f'{dataset_name}_{model}_{log_type}', arrival_times)
            arrival_times_obj.save()
            logger.info(f"Finished processing all requests for {dataset_name} {model} {log_type}.")
        
    elif dataset_name == 'humaneval':
        dataset = datasets.load_dataset('openai/openai_humaneval')
        dataset = dataset['train']
        logger.info(f"Loaded dataset with {len(dataset)} examples.")
        
        # Process requests in parallel
        import concurrent.futures
        def create_humaneval_request(ex):
            return Request(
                input_length=ex['input_length'],
                output_length=ex['output_length']
            )
        
        logger.info("Creating Request objects in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            request_futures = [executor.submit(create_humaneval_request, ex) for ex in dataset]
            
            for idx, future in enumerate(concurrent.futures.as_completed(request_futures)):
                if idx % 1000 == 0 and idx > 0:
                    logger.info(f"Processed {idx} requests so far...")
                requests.append(future.result())
        logger.info(f"Finished processing all requests for {dataset_name}.")
    
    elif dataset_name == 'arxiv_summary':
        logger.info("Loading 'arxiv_summary' dataset...")
        dataset = datasets.load_dataset('ccdv/arxiv-summarization')
        dataset = dataset['train']
        dataset = dataset.select(range(10000))
        logger.info(f"Loaded dataset with {len(dataset)} examples.")
        
        # Process requests in parallel
        import concurrent.futures
        def create_arxiv_request(ex):
            return Request(
                prompt=ex['article'],
                answer=ex['abstract'],
                input_length=count_length(ex['article']),
                output_length=count_length(ex['abstract'])
            )
        
        logger.info("Creating Request objects in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            request_futures = [executor.submit(create_arxiv_request, ex) for ex in dataset]
            
            for idx, future in enumerate(concurrent.futures.as_completed(request_futures)):
                if idx % 1000 == 0 and idx > 0:
                    logger.info(f"Processed {idx} requests so far...")
                requests.append(future.result())
        logger.info(f"Finished processing all requests for {dataset_name}.")
    else: 
        logger.error(f'Dataset {dataset_name} not found')
        raise ValueError(f'Dataset {dataset_name} not found')
    logger.info(f'{dataset_name} has {len(requests)} requests')
    requests_obj = Requests(dataset_name, requests)
    logger.info(f"Saving requests object for {dataset_name}...")
    requests_obj.save()
    logger.info(f"Requests object saved for {dataset_name}.")
    
if __name__ == '__main__':
    import asyncio
    import concurrent.futures
    
    def download_parallel(dataset_name):
        try:
            download_dataset(dataset_name)
            logger.info(f"Successfully downloaded {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
    
    datasets_to_download = [
        'azure_code',
        'azure_chat', 
        'azure_chat_23',
        'azure_code_23',
        # 'sharegpt_chat',
        # 'sharegpt_code',
        # 'burstgpt',
        # 'humaneval',
        # 'arxiv_summary',
        # 'deepseek-r1'
    ]
    
    logger.info(f"Starting parallel download of {len(datasets_to_download)} datasets...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(download_parallel, dataset) for dataset in datasets_to_download]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Dataset download failed: {e}")
            
    
    logger.info("All dataset downloads completed!")