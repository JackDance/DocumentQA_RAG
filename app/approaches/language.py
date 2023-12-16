#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：beigene-demo
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/4/13 10:27 
# ====================================
import asyncio
import json
import time
import uuid
import aiohttp
from app.azureai import chat_completion
from .approach import Approach
from app.exceptions import TranslatorException


class LanguageDetectApproach(Approach):

    async def detect_language(self, text):
        prompt = f"""
        检测以下文本的语言类型,直接输出输入文本的语言类型

        examples:
        =========================
        input: 请解释什么是人工智能？
        output: Chinese
        =========================
        input: Please explain what is artificial intelligence?
        output: English
        =========================
        input: {text}
        output:
        """
        messages = [
            {"role": "user", "content": prompt}
        ]

        start = time.time()
        start = time.time()
        completion = await chat_completion(
            messages, model="chat", max_tokens=10, stop=["\n", ".", "。"]
        )
        print(f"Detect time:{time.time() - start}")
        return completion

    async def run(self, history: list[dict], overrides: dict, model) -> any:
        question = history[-1]["user"]
        # 检测语言
        lang = await self.detect_language(question)
        return lang


class AzureLanguageDetectApproach(Approach):
    """
    使用 Azure translator 检测语言
    """

    def __init__(self, translator_endpoint: str, translator_key: str):
        self.translator_endpoint = translator_endpoint
        self.translator_key = translator_key

    @property
    def name(self):
        return "Azure Translator"

    @staticmethod
    def postpreprocess(language):
        lang_map = {
            "zh-Hans": "Chinese",
            "en": "English"
        }

        return lang_map.get(language, "Chinese")

    async def detect(self, question):
        body = [{'text': question}]
        params = {'api-version': '3.0'}
        headers = {
            'Ocp-Apim-Subscription-Key': self.translator_key,
            'Ocp-Apim-Subscription-Region': "westus2",
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }

        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(
                        self.translator_endpoint + '/detect',
                        params=params,
                        headers=headers,
                        data=json.dumps(body)) as response:
                    res = await response.json()
                    lang = res[0]["language"]
                    return self.postpreprocess(lang)
            except asyncio.TimeoutError:
                raise TranslatorException(name="azure translator")
            except aiohttp.ClientConnectorError:
                raise TranslatorException(name="azure translator")

    async def run(self, history: list[dict], overrides: dict, model) -> any:
        question = history[-1]["user"]
        # 检测语言
        lang = await self.detect(question)
        print(f"{self.name} Detect Language: {lang}")
        return lang
