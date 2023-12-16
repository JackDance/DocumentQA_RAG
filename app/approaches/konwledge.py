#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：beigene-demo
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/4/13 10:27 
# ====================================
import os.path
import time
from typing import List

import tiktoken
from app.azureai import chat_completion, AzureOpenAI
from .approach import Approach
from langchain.vectorstores import Milvus


class KnowledgeApproach(Approach):
    prompt_prefix = """<|im_start|>system
                    作为AI助手“百济星”，基于知识库中的内容回答。
                    生成的答案必须从知识库中的内容提取，不能依赖任何先验知识回答，如果信息不足，请根据用户的语言回复“我不知道”或要求用户提供更多信息
                    聊天历史记录的信息仅作为聊天上下文的参考，不作为回答问题的依据。
                    始终包括您在生成的答案中使用的每个事实的来源名称，引用来源(例如:[info.pdf#page=1][智能免疫助手QnA.txt#page=10])。
                    {markdown_table_prompt}
                    {follow_up_questions_prompt}
                    {injected_prompt}
                    
                    知识库：
                    {sources}
                    
                    对话：
                    {chat_history}
                    
                    回答时，您必须使用以下语言：{language}。
                    答案格式：答案内容 [引用1][引用2] <<后续问题1>><<后续问题2>><<后续问题3>>
                    """

    # follow up question
    follow_up_questions_prompt = """生成最多三个带有双尖括号的用户可能会问的后续问题。
                                尽量不要重复已经问过的问题。
                                只生成问题，避免使用停用词。
                                只生成问题，不在问题前后生成任何文本，例如“接下来可能的问题”
                                后续问题示例：
                                <<百济神州的创始人>><<百济在哪里上市的>><<百济的商业化产品有哪些>>
                                后续问题不能是下面的这个：
                                <<后续问题1>><<后续问题2>><<后续问题3>>
                                """
    # 以Markdown显示表格信息,生成表格的单元格中不要出现换行符“\n”,如果出现“\n”,将其替换成Html的换行标记“<br>”
    # markdown 2 html
    markdown_table_prompt = """生成的答案中能生成表格的都严格生成markdown表格，不要生成Markdown列表格式
                        如果出现列表格式，将其转换成Markdown表格。
                        Markdown表格示例(生成的表格必须严格按照示例格式)：
                        
                        | 列1 | 列2 | 列3 |
                        | --- | --- | --- |
                        | 行1单元格1 | 行1单元格2 | 行1单元格3 |
                        | 行2单元格1 | 行2单元格2 | 行2单元格3 |
                        
                        生成表格的单元格中不要出现换行符“\n”,如果出现“\n”,将其替换成Html的换行标记“<br>”"""

    def __init__(
            self,
            knowledge_datastore: Milvus,
            azure_ai_endpoint: AzureOpenAI):
        self.knowledge_datastore = knowledge_datastore
        self.azure_ai_endpoint = azure_ai_endpoint
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_length = 4000
        self.min_length = 200
        self.faq_threshold = 0.3

    def length_function(self, text):
        return len(self.tokenizer.encode(text))

    async def step(self, history, overrides, model, sub_question=None):
        # 问题
        question = sub_question or history[-1]["user"]

        print(f"Question: {question}")
        # 检测语言
        lang = overrides["language"]

        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_template")
        if prompt_override is None:
            injected_prompt = ""
        elif prompt_override.startswith(">>>"):
            injected_prompt = prompt_override.replace(">>>", "")
        else:
            injected_prompt = ""
        suggest_followup_questions = overrides.get("suggest_followup_questions", None)
        if suggest_followup_questions:
            follow_up_questions_prompt = self.follow_up_questions_prompt
        else:
            follow_up_questions_prompt = ""

        # 聊天历史记录
        chat_history_str = self.get_chat_history_as_text(history, include_last_turn=False, approx_max_tokens=1000)
        chat_history_str += f"<|im_start|>user\n{question}\n<|im_start|>assistant\n"
        chat_history_len = self.length_function(chat_history_str)

        # prompt len
        prompt_len = self.length_function(
            self.prompt_prefix + injected_prompt + follow_up_questions_prompt + self.markdown_table_prompt)

        start = time.time()
        # Knowledge
        result_with_score = self.knowledge_datastore.similarity_search_with_score(question, k=15)

        sources_max_len = self.max_length - prompt_len - chat_history_len

        print("Embedding time:", time.time() - start)
        # 构建sources上下文
        knowledge_data_points = []

        if result_with_score:
            for document, score in result_with_score:
                # for document in result_with_score:
                page_content = document.page_content
                page = document.metadata["page"] + 1
                source = document.metadata["source"].replace("\\", "/")
                source = os.path.basename(source)

                # 过滤比较小的段落
                # if self.length_function(page_content) < self.min_length:
                #     continue

                # 拼接合适长度的段落
                file_name_prefix, file_type_suffix = os.path.splitext(source)
                data_point = f"{file_name_prefix}{file_type_suffix}#page={page}: {page_content}"
                current_len = self.length_function(str(knowledge_data_points) + data_point)
                if current_len < sources_max_len:
                    knowledge_data_points.append(data_point)

        sources_str = "\n".join(knowledge_data_points)
        prompt_prefix = self.prompt_prefix
        markdown_table_prompt = self.markdown_table_prompt

        # prompt
        prompt = prompt_prefix.format(injected_prompt=injected_prompt,
                                      sources=sources_str,
                                      chat_history=chat_history_str,
                                      follow_up_questions_prompt=follow_up_questions_prompt,
                                      markdown_table_prompt=self.markdown_table_prompt,
                                      language=lang
                                      )

        messages = [
            {"role": "user", "content": prompt}
        ]

        start = time.time()
        completion = await self.azure_ai_endpoint.chat_completion(
            messages, model=model, max_tokens=4000, stop=["<|im_end|>", "<|im_start|>", "<|", " <|"]
        )
        print("Prompt len:", self.length_function(prompt))
        print("Chat time:", time.time() - start)
        print("Completion:", completion)

        datapoint = knowledge_data_points
        thought = f"Searched for:<br>{question}<br><br>Prompt:<br>" + prompt.replace('\n', '<br>')
        answer = completion
        return datapoint, answer, thought

    async def run(self, history: list[dict], overrides: dict, model: str, sub_question_list: List[str] = None) -> any:
        knowledge_data_points = []
        answers = []
        thoughts = []
        for sub_question in sub_question_list:
            datapoint, answer, thought = await self.step(history, overrides, model, sub_question)
            knowledge_data_points.extend(datapoint)
            answers.append(answer)
            thoughts.append(thought)

        r = {
            "data_points": knowledge_data_points,
            "answer": "<br>".join(answers),
            "thoughts": f"Model Name: {model} <br>" + "<br>".join(thoughts)
        }

        return r

    # 修改为流式输出形式
    async def run_streaming(self, history: list[dict], overrides: dict, model: str, sub_question_list: List[str] = None,
                  file_id=None) -> any:
        knowledge_data_points = []
        source2adal_urls = {}
        answers = []
        thoughts = []
        question = sub_question_list[-1]
        print(f"File_id: {file_id}")
        print(f"Question: {question}")
        # 检测语言
        lang = overrides["language"]

        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_template")
        if prompt_override is None:
            injected_prompt = ""
        elif prompt_override.startswith(">>>"):
            injected_prompt = prompt_override.replace(">>>", "")
        else:
            injected_prompt = ""
        suggest_followup_questions = overrides.get("suggest_followup_questions", None)
        if suggest_followup_questions:
            follow_up_questions_prompt = self.follow_up_questions_prompt
        else:
            follow_up_questions_prompt = ""

        # 聊天历史记录
        chat_history_str = self.get_chat_history_as_text(history, include_last_turn=False, approx_max_tokens=1000)
        chat_history_str += f"<|im_start|>user\n{question}\n<|im_start|>assistant\n"
        chat_history_len = self.length_function(chat_history_str)

        # prompt len
        prompt_len = self.length_function(
            self.prompt_prefix + injected_prompt + follow_up_questions_prompt + self.markdown_table_prompt)

        start = time.time()

        expr = f"file_id in [{file_id}]" if file_id else None

        source2adal_url = {}

        # Knowledge
        result_with_score = self.knowledge_datastore.similarity_search_with_score(question, k=10, expr=expr)
        print(f"result_with_score: {result_with_score}")
        sources_max_len = self.max_length - prompt_len - chat_history_len

        print("Embedding time:", time.time() - start)

        if result_with_score:
            for document, score in result_with_score:
                # for document in result_with_score:
                page_content = document.page_content
                page = document.metadata["page"] + 1
                source = document.metadata["source"].replace("\\", "/")
                source = os.path.basename(source)

                # 拼接合适长度的段落
                file_name_prefix, file_type_suffix = os.path.splitext(source)
                data_point = f"{file_name_prefix}{file_type_suffix}#page={page}: {page_content}"
                current_len = self.length_function(str(knowledge_data_points) + data_point)
                if current_len < sources_max_len:
                    knowledge_data_points.append(data_point)

        sources_str = "\n".join(knowledge_data_points)
        prompt_prefix = self.prompt_prefix
        markdown_table_prompt = self.markdown_table_prompt

        # prompt
        prompt = prompt_prefix.format(injected_prompt=injected_prompt,
                                      sources=sources_str,
                                      chat_history=chat_history_str,
                                      follow_up_questions_prompt=follow_up_questions_prompt,
                                      markdown_table_prompt=self.markdown_table_prompt,
                                      language=lang
                                      )

        print("Prompt len:", self.length_function(prompt))
        print(f"Chat time: {round((time.time() - start), 2)}s")

        datapoint = knowledge_data_points
        thought = f"Searched for:<br>{question}<br><br>Prompt:<br>" + prompt.replace('\n', '<br>')
        knowledge_data_points.extend(datapoint)
        thoughts.append(thought)
        source2adal_urls.update(source2adal_url)

        res = {
            "datapoints": knowledge_data_points,
            "thoughts": f"Model Name: {model} <br>" + "<br>".join(thoughts),
            "source2adal_urls": source2adal_urls,
            "prompt": prompt
        }
        return res

    def get_chat_history_as_text(self, history, include_last_turn=True, approx_max_tokens=1000) -> str:
        history_text = ""
        for h in reversed(history if include_last_turn else history[:-1]):
            # 当前会话
            curr_conversation = """<|im_start|>user""" + "\n" + h[
                "user"] + "\n" + """<|im_end|>""" + "\n" + """<|im_start|>assistant""" + "\n" + (
                                    h.get("bot") + """<|im_end|>""" if h.get("bot") else "") + "\n"
            # 计算 token
            if self.length_function(history_text + curr_conversation) > approx_max_tokens:
                break
            else:
                history_text = curr_conversation + history_text
        return history_text
