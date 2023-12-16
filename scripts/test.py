# !/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：beigene-demo
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/4/12 9:49
# ====================================
import numpy as np
import openai
import pandas as pd

# openai.api_key = "2e3bc8a0624246b9b5a679f11b5ce5cb"
# openai.api_type = "azure"
# openai.api_base = "https://apac-openai-service-eus.openai.azure.com/"
# openai.api_version = "2023-03-15-preview"

openai.api_key = "dfc23a65cf9e40cea71911cf93ca444e"
openai.api_type = "azure"
openai.api_base = "https://apac-openai-service-fc.openai.azure.com/"
openai.api_version = "2023-03-15-preview"

# openai.api_key = "afc31eb73f474752b5ea7e1a4613b4a8"
# openai.api_type = "azure"
# openai.api_base = "https://apac-openai-service-scus.openai.azure.com/"
# openai.api_version = "2023-03-15-preview"


#
# question = "haha"
# response = openai.Embedding.create(input=question, engine='text-embedding-ada-002')
#
# print(response['data'][0]['embedding'])



# model = "gpt4"
model = "gpt35turbo"

messages = [
    {"role": "user", "content": "你的基座模型是什么"}
]

# call the OpenAI chat completion API with the given messages
response = openai.ChatCompletion.create(
    engine=model,
    messages=messages,
    temperature=0,
    max_tokens=1024,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None
)

choices = response["choices"]  # type: ignore
completion = choices[0].message.content.strip()
print(completion)
