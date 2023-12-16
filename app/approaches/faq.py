#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：beigene-demo
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/4/13 10:27 
# ====================================
import json
import random
from typing import List

from app.azureai import chat_completion
from .approach import Approach
from langchain.vectorstores import Milvus


class FAQApproach(Approach):

    def __init__(
            self,
            datastore: Milvus):
        self.datastore = datastore
        self.max_rounds = 2

    def check_round_over(self, faqs, question):
        prompt = """检测当前的FAQ问答对是否能够完全回答问题
                    问题中可能会包含多个问题组合在一起的情况
                    判断是否能够完全回答问题。如果可以完全回答问题，输出"True"，否则输出"False"。
        
                    FAQ问答对：
                    {faqs}

                    问题：
                    {question}
                    """

        prompt = prompt.format(faqs=faqs, question=question)
        messages = [
            {"role": "user", "content": prompt}
        ]
        completion = chat_completion(
            messages, model="chat", max_tokens=10, stop=["\n", ".", "。"]
        )
        return eval(completion)

    def step(self, history, overrides, id_records):
        # 问题
        question = history[-1]["user"]

        history_question = [h["user"] for h in history[:-1]]
        suggest_followup_questions = overrides.get("suggest_followup_questions", None)
        k = 10 if suggest_followup_questions else 1
        # result_with_score = self.datastore.similarity_search_with_score(question, k=k)
        if id_records:
            expr = f"id not in {id_records}"
        else:
            expr = ""
        _, result = self.datastore._worker_search(
            question, k=k, param=None, expr=expr
        )

        follow_up = []
        document, score, id = result[0]
        if score < 0.3:
            status = True
            faq_question = document.page_content
            faq_answer = document.metadata["answer"]
            faq = {"question": faq_question, "answer": faq_answer}
            id_records.append(id)
            if suggest_followup_questions:
                follow_up = [f"<<{d.page_content}>>" for d, s, _ in result[1:] if
                             d.page_content not in history_question]
        else:
            status = False
            faq = {"question": "", "answer": ""}

        return status, faq, follow_up, id_records

    def run(self, history: list[dict], overrides: dict) -> any:
        # 问题
        question = history[-1]["user"]
        answers = []
        faqs = []
        follow_up_questions = []
        id_records = []

        status_list = []
        for i in range(self.max_rounds):
            status, faq, follow_up, id_records = self.step(history, overrides, id_records)
            status_list.append(status)
            faqs.append(f"问题：{faq['question']}\n答案：{faq['answer']}")
            follow_up_questions.extend(follow_up)
            answers.append(faq['answer'])
            over_flag = self.check_round_over(faqs, question)
            if over_flag:
                break
        follow_up_questions = random.sample(follow_up_questions, 3) if len(
            follow_up_questions) >= 3 else follow_up_questions
        result = {"data_points": [], "answer": "\n".join(answers) + "".join(follow_up_questions), "thoughts": ""}

        return any(status_list), result

    def insert(self, question: str, answer: dict):
        result = self.datastore.add_texts([question], metadatas=[{"answer": json.dumps(answer, ensure_ascii=False)}])
        print(result)


class FAQPlanApproach(Approach):

    def __init__(
            self,
            datastore: Milvus):
        self.datastore = datastore
        self.max_rounds = 2

    def step(self, history, overrides, id_records, sub_question=None):
        # 问题
        question = sub_question or history[-1]["user"]

        history_question = [h["user"] for h in history[:-1]]
        suggest_followup_questions = overrides.get("suggest_followup_questions", None)
        k = 10 if suggest_followup_questions else 1
        # result_with_score = self.datastore.similarity_search_with_score(question, k=k)
        if id_records:
            expr = f"id not in {id_records}"
        else:
            expr = ""
        _, result = self.datastore._worker_search(
            question, k=k, param=None, expr=expr
        )

        follow_up = []
        document, score, id = result[0]
        if score < 0.1:
            status = True
            faq_question = document.page_content
            faq_answer = document.metadata["answer"]
            faq = {"question": faq_question, "answer": faq_answer}
            id_records.append(id)
            if suggest_followup_questions:
                follow_up = [f"<<{d.page_content}>>" for d, s, _ in result[1:] if
                             d.page_content not in history_question]
        else:
            status = False
            faq = {"question": "", "answer": ""}

        return status, faq, follow_up, id_records

    def run(self, history: list[dict], overrides: dict, model: str, sub_question_list: List[str] = None
            ) -> any:
        answers = []
        faqs = []
        follow_up_questions = []
        id_records = []
        status_list = []

        for sub_question in sub_question_list:
            status, faq, follow_up, id_records = self.step(history, overrides, id_records, sub_question=sub_question)
            status_list.append(status)
            faqs.append(f"问题：{faq['question']}\n答案：{faq['answer']}")
            follow_up_questions.extend(follow_up)
            answers.append(faq['answer'])

        follow_up_questions = random.sample(follow_up_questions, 3) if len(
            follow_up_questions) >= 3 else follow_up_questions

        sub_question_str = '<br>'.join(sub_question_list)
        result = {
            "data_points": [],
            "answer": "<br>".join(answers) + "".join(follow_up_questions),
            "thoughts": f"Searched for:<br>{sub_question_str}<br>"
        }

        return any(status_list), result

    def insert(self, question: str, answer: dict):
        result = self.datastore.add_texts([question], metadatas=[{"answer": json.dumps(answer, ensure_ascii=False)}])
        print(result)
