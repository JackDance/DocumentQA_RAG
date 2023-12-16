#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：beigene-async
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/6/29 18:16 
# ====================================
import asyncio
import json
import logging
import random
from typing import Dict, List, Tuple
import aiohttp
from tenacity import wait_random_exponential, retry, stop_after_attempt

from app.exceptions import OpenAITimeoutException


# @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(2))
async def post(
        uri: str,
        headers: Dict = {},
        data: Dict = {}
):
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(uri, headers=headers, data=json.dumps(data)) as response:
                res = await response.json()
                return res
        except asyncio.TimeoutError:
            raise OpenAITimeoutException(name="azureai")
        except aiohttp.ClientConnectorError:
            raise OpenAITimeoutException(name="azureai")


async def embedding(text):
    # Call the OpenAI API to get the embeddings
    uri = "https://apac-openai-service-eus.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-03-15-preview"

    openai_api_key = "2e3bc8a0624246b9b5a679f11b5ce5cb"
    headers = {"api-key": f"{openai_api_key}", "Content-Type": "application/json"}
    data = {"input": text}

    json_res = await post(uri, headers=headers, data=data)

    return json_res['data'][0]['embedding']


async def chat_completion(
        messages,
        model="chat",
        max_tokens=512,
        temperature=0,
        stop=None
):
    uri = f"https://apac-openai-service-eus.openai.azure.com/openai/deployments/{model}/chat/completions?api-version=2023-03-15-preview"
    openai_api_key = "2e3bc8a0624246b9b5a679f11b5ce5cb"
    # 请求头
    headers = {"api-key": f"{openai_api_key}", "Content-Type": "application/json"}

    # 请求体
    data = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "n": 1
    }

    completion = await post(uri, headers=headers, data=data)
    print("Input Completion: ", completion)
    # 由于openai的安全机制，若不符合安全机制，则将content置为空
    if completion["choices"][0]["finish_reason"] == "content_filter":
        return "[]"
    else:
        return completion["choices"][0]["message"]["content"].strip()


class AzureOpenAI(object):

    def __init__(
            self,
            embedding_endpoints: List[Tuple],
            gpt35_chat_endpoints: List[Tuple],
            gpt4_chat_endpoints: List[Tuple],
            max_retry: int = 2
    ):
        self.embedding_endpoints = embedding_endpoints
        self.gpt35_chat_endpoints = gpt35_chat_endpoints
        self.gpt4_chat_endpoints = gpt4_chat_endpoints
        self.max_retry = max_retry

    @staticmethod
    async def post(
            uri: str,
            headers: Dict = {},
            data: Dict = {}
    ):
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(uri, headers=headers, data=json.dumps(data)) as response:
                    res = await response.json()
                    return True, res
            except asyncio.TimeoutError:
                return False, None
            except aiohttp.ClientConnectorError:
                return False, None

    async def embedding(self, text):

        endpoint_records = []

        for i in range(self.max_retry):
            if i > 0:
                ready_endpoints = list(set(self.embedding_endpoints) - set(endpoint_records))
            else:
                ready_endpoints = self.embedding_endpoints
            endpoint = random.choice(ready_endpoints)

            endpoint_uri, key, emb_model_name = endpoint

            uri = endpoint_uri + f"/openai/deployments/{emb_model_name}/embeddings?api-version=2023-03-15-preview"
            headers = {"api-key": key, "Content-Type": "application/json"}
            data = {"input": text}

            status, json_res = await self.post(uri, headers=headers, data=data)
            endpoint_records.append(endpoint)
            if status:
                return json_res['data'][0]['embedding']
        raise OpenAITimeoutException("openai")

    async def chat_completion(
            self,
            messages,
            model="chat",
            max_tokens=512,
            temperature=0,
            stop=None
    ):

        endpoint_records = []
        # 法国区 deployment model name 和 美区不一致
        if model == "chat":
            chat_endpoints = self.gpt35_chat_endpoints
        else:
            chat_endpoints = self.gpt4_chat_endpoints

        for i in range(self.max_retry):
            if i > 0:
                ready_endpoints = list(set(chat_endpoints) - set(endpoint_records))
            else:
                ready_endpoints = chat_endpoints
            endpoint = random.choice(ready_endpoints)

            endpoint_uri, key, chat_model_name = endpoint

            # 法国区 deployment model name 和 美区不一致

            print(f"Call Azure OpenAI Chat endpoint: {endpoint_uri}")

            uri = endpoint[
                      0] + f"/openai/deployments/{chat_model_name}/chat/completions?api-version=2023-03-15-preview"
            headers = {"api-key": key, "Content-Type": "application/json"}
            # 请求体
            data = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
                "n": 1
            }

            status, json_res = await self.post(uri, headers=headers, data=data)
            endpoint_records.append(endpoint)
            if status and "choices" in str(json_res):
                return json_res["choices"][0]["message"]["content"].strip()
        raise OpenAITimeoutException("openai")


if __name__ == '__main__':
    # loop = asyncio.get_event_loop()
    # res = loop.run_until_complete(embedding("hello"))
    # print(res)

    embedding_endpoints = [
        # (endpoint, key, model_name)
        ("https://apac-openai-service-eus.openai.azure.com", "2e3bc8a0624246b9b5a679f11b5ce5cb",
         "text-embedding-ada-002"),
        ("https://apac-openai-service-scus.openai.azure.com", "afc31eb73f474752b5ea7e1a4613b4a8",
         "text-embedding-ada-002"),
    ]

    gpt35_chat_endpoints = [
        # (endpoint, key, model_name)
        ("https://apac-openai-service-eus.openai.azure.com", "2e3bc8a0624246b9b5a679f11b5ce5cb", "chat"),
        ("https://apac-openai-service-scus.openai.azure.com", "afc31eb73f474752b5ea7e1a4613b4a8", "chat"),
        ("https://apac-openai-service-fc.openai.azure.com", "dfc23a65cf9e40cea71911cf93ca444e", "gpt35turbo")
    ]

    gpt4_chat_endpoints = [
        # (endpoint, key, model_name)
        ("https://apac-openai-service-eus.openai.azure.com", "2e3bc8a0624246b9b5a679f11b5ce5cb", "gpt4"),
        ("https://apac-openai-service-fc.openai.azure.com", "dfc23a65cf9e40cea71911cf93ca444e", "gpt4")
    ]
    # Azure ai Endpoint
    azure_ai_endpoint = AzureOpenAI(embedding_endpoints, gpt35_chat_endpoints, gpt4_chat_endpoints)

    loop = asyncio.get_event_loop()
    # res = loop.run_until_complete(azure_ai_endpoint.embedding("hello"))

    messages = [
        {"role": "user", "content": "hello"}
    ]
    res = loop.run_until_complete(azure_ai_endpoint.chat_completion(messages, model="gpt4"))
    print(res)
