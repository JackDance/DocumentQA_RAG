#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：beigene-demo
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/4/13 10:27 
# ====================================

from app.azureai import chat_completion
from .approach import Approach


class QuestionDecomposeApproach(Approach):

    async def question_decompose(self, question, model):
        prompt = """问题中可能会包含多个问题组合在一起的情况，把问题拆分成子问题。
        如果无法拆分，请将原始问题作为子问题。
        examples:
        ---
        question: 百济神州的创始人
        sub_question: ["百济神州的创始人"]
        ___
        question: 来那度胺耐药、硼替佐米耐药MM是如何定义的？如何使用问答库？
        sub_question: ["来那度胺耐药、硼替佐米耐药MM是如何定义的？","如何使用问答库？"]
        ---
        question: {question}
        sub_question: 
        """
        prompt = prompt.format(question=question)
        messages = [
            {"role": "user", "content": prompt}
        ]
        completion = await chat_completion(
            messages,
            model=model,
            max_tokens=512
        )
        return eval(completion)

    async def run(self, history: list[dict], overrides: dict, model: str) -> any:
        # 原始问题
        question = history[-1]["user"]
        # 多个问题拆解为子问题
        sub_question_list = await self.question_decompose(question, model)

        return sub_question_list
