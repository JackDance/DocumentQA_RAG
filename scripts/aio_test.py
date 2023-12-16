#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：beigene-async
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/6/30 13:32 
# ====================================
import aiohttp
import asyncio
import json


async def main():
    uri = f"https://apac-openai-service-eus.openai.azure.com/openai/deployments/chat/chat/completions?api-version=2023-03-15-preview"
    openai_api_key = "2e3bc8a0624246b9b5a679f11b5ce5cb"
    prompt = "Once upon a time"

    messages = [
        {"role": "user", "content": prompt}
    ]

    headers = {"api-key": f"{openai_api_key}", "Content-Type": "application/json"}

    data = {
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0,
        "stop": None,
        "n": 1
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(uri, headers=headers, data=json.dumps(data)) as response:
            print("Status:", response.status)
            print("Content-type:", response.headers['content-type'])

            html = await response.json()
            print("Body:", html, "...")


async def get_embedding(text):
    """
    Embed texts using OpenAI's ada model.

    Args:
        texts: The list of texts to embed.

    Returns:
        A list of embeddings, each of which is a list of floats.

    Raises:
        Exception: If the OpenAI API call fails.
    """
    # Call the OpenAI API to get the embeddings
    uri = "https://apac-openai-service-eus.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-03-15-preview"
    openai_api_key = "2e3bc8a0624246b9b5a679f11b5ce5cb"
    headers = {"api-key": f"{openai_api_key}", "Content-Type": "application/json"}

    data = {
        "input": text
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(uri, headers=headers, data=json.dumps(data)) as response:
            print("Status:", response.status)
            print("Content-type:", response.headers['content-type'])

            html = await response.json()
            print("Body:", html, "...")


loop = asyncio.get_event_loop()
loop.run_until_complete(get_embedding("hello"))
# loop.run_until_complete(main())
