#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：beigene-chunk
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/5/25 11:02 
# ====================================

import json
import re
import sys

import langchain
import os
from typing import List, Tuple
import openai
import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
from pymilvus import FieldSchema, DataType, CollectionSchema
from store import MilvusCollection

openai.api_key = "2e3bc8a0624246b9b5a679f11b5ce5cb"
openai.api_type = "azure"
openai.api_base = "https://apac-openai-service-eus.openai.azure.com/"
openai.api_version = "2023-03-15-preview"


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def refine(question, answer):
    prompt = f"""
    下面是faq问答对，帮我重写答案
    examples:
    =========================
    question：腹泻
    answer：[链接](https://mp.weixin.qq.com/s?__biz=Mzg4OTU4NTIzMw==&mid=2247488285&idx=1&sn=aeb0340f311042b6bf1cad349449f8c6&chksm=cfe8f823f89f7135fc6227fafac3ac0d473f1f36d05081790e380c7a55f400a864ba26c93196#rd)
    refine_answer: 根据您提供的信息，我找到了一篇公众号推文，提供关于针对腹泻的处理办法。您可以点击以下链接了解更多详情：[链接](https://mp.weixin.qq.com/s?__biz=Mzg4OTU4NTIzMw==&mid=2247488285&idx=1&sn=aeb0340f311042b6bf1cad349449f8c6&chksm=cfe8f823f89f7135fc6227fafac3ac0d473f1f36d05081790e380c7a55f400a864ba26c93196#rd)
    =========================
    question: {question}
    answer: {answer}
    refine_answer: 
    """

    messages = [
        {"role": "user", "content": prompt}
    ]

    completion = get_chat_completion(
        messages, model="chat", max_tokens=1024
    )
    return completion


def create_schema():
    """
    Returns the schema of the collection.

    :return: The schema of the collection.
    :rtype: CollectionSchema
    """
    id = FieldSchema(
        name="id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=True
    )
    question = FieldSchema(
        name="question",
        dtype=DataType.VARCHAR,
        max_length=200,
    )
    answer = FieldSchema(
        name="answer",
        dtype=DataType.VARCHAR,
        max_length=6000,
    )
    embedding = FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=1536
    )
    schema = CollectionSchema(
        fields=[id, question, embedding, answer],
        description="faq schema",
        auto_id=True
    )
    return schema


def excel2txt():
    # 内部文件列表
    internal_file_list = [
        "凯洛斯问答机器人.xlsx",
        "安加维百科.xlsx",
        "百汇泽百科全书.xlsx"
    ]
    # 外部文件
    external_file_list = ["千方百济.xlsx"]

    # 数据
    data = {}

    # 处理 智能免疫助手QnA.txt 文件
    with open("../data/raw_faq/智能免疫助手QnA.txt", "r", encoding="utf8") as f:

        curr_container = []
        curr_text = ""
        for d in f.readlines():
            if d.startswith("问题："):
                curr_text += d.strip()
            elif d.startswith("相似问题："):
                curr_text += f"\n{d.strip()}"
            elif d.startswith("回答："):
                a = d.strip().replace("回答：", "答案：")
                curr_text += f"\n{a}"
                curr_container.append(curr_text)
                curr_text = ""
    data["智能免疫助手QnA.txt"] = curr_container

    excel_files: List[str] = [os.path.join("../data/raw_faq", f) for f in os.listdir("../data/raw_faq") if
                              f in external_file_list]
    for e in excel_files:
        k_df = pd.read_excel(e, engine="openpyxl")
        question_column = "基础问题"
        answer_column = "微信公众号推文链接"
        sim_question_columns = [i for i in k_df.columns if i.startswith("关联问题")]
        k_df.fillna("0", inplace=True)
        for ix, row in k_df.iterrows():
            if row[question_column] == "0":
                continue
            qa = ""
            question = f"问题：{row[question_column]}"
            sim_question = [x for x in row[sim_question_columns].tolist() if x != "0"]
            answer = re.sub('\n+', '\n', row[answer_column].strip())
            answer = f"答案：根据您提供的问题，我找到了一篇公众号推文,点击链接了解详情：[链接]({answer})"
            if sim_question:
                sim_q_str = '\n'.join([f"相似问题：{x}" for x in sim_question])
                qa += f"{question}\n{sim_q_str}\n{answer}"

            else:
                qa += f"{question}\n{answer}"

            to_file = f"{os.path.splitext(os.path.basename(e))[0]}.txt"
            if to_file not in data.keys():
                data[to_file] = [qa]
            else:
                data[to_file].append(qa)

    excel_files: List[str] = [os.path.join("../data/raw_faq", f) for f in os.listdir("../data/raw_faq") if
                              f in internal_file_list]
    for e in excel_files:
        k_df = pd.read_excel(e, engine="openpyxl")
        question_column = "标准问题(80字以内，不能为空)"
        answer_column = "答案(2000字以内，不能为空)"
        sim_question_columns = [i for i in k_df.columns if i.startswith("相似问法")]
        k_df.fillna("0", inplace=True)
        for ix, row in k_df.iterrows():
            if row[question_column] == "0":
                continue
            qa = ""
            question = f"问题：{row[question_column]}"
            sim_question = [x for x in row[sim_question_columns].tolist() if x != "0"]
            answer = re.sub('\n+', '\n', row[answer_column].strip())
            answer = f"""答案：{answer}"""
            if sim_question:
                sim_q_str = '\n'.join([f"相似问题：{x}" for x in sim_question])
                qa += f"{question}\n{sim_q_str}\n{answer}"

            else:
                qa += f"{question}\n{answer}"
            to_file = f"{os.path.splitext(os.path.basename(e))[0]}.txt"
            if to_file not in data.keys():
                data[to_file] = [qa]
            else:
                data[to_file].append(qa)

    return data


# 删除 使用 ChatGPT 优化faq 中的答案
def refine_answer():
    k_df = pd.read_excel("data/raw_faq/千方百济.xlsx", engine="openpyxl")

    data = []

    question_column = "基础问题"
    answer_column = "微信公众号推文链接"
    sim_question_columns = [i for i in k_df.columns if i.startswith("关联问题")]
    k_df.fillna("0", inplace=True)
    for ix, row in k_df.iterrows():
        q_list = [row[question_column]] + row[sim_question_columns].tolist()
        q_list = [i for i in q_list if i != "0"]
        a = row[answer_column]
        for q in q_list:
            try:

                ref_a = refine(q, a)
            except Exception as e:
                print(e)
                ref_a = refine(q, a)
            row = {"question": q, "answer": ref_a}
            data.append(row)
            print(q)
            print(a)
            print(ref_a)
            print("=============================================")

    df = pd.DataFrame(data)
    df.to_csv("data/raw_faq/千方百济.csv", index=False)


# 1. raw_faq 文件转换为 txt 文件（统一指定格式）
def raw_convert_txt():
    data = excel2txt()

    for filename, row in data.items():
        with open(f"data/faq/{filename}", "w", encoding="utf8") as f:
            write_data = "\n\n".join([i for i in row])
            f.write(write_data)


# 2. 导入embedding database
def insert_embedding_db():
    # files: List[str] = [os.path.join("data/faq", f) for f in os.listdir("data/faq") if f.endswith(".txt")]
    files: List[str] = [os.path.join("../data/faq", f) for f in os.listdir("../data/faq") if f.endswith("替雷利珠单抗蛋白质序列.txt")]

    chunks_data = []

    # 读取数据
    for f in files:
        with open(f, "r", encoding="utf8") as read_f:
            read_data = read_f.read()
            lines = read_data.splitlines()
            line_map = {l: ix for ix, l in enumerate(lines)}
            data = read_data.split("\n\n")
            for d in data:
                question = d.split("\n")[0].strip()
                row = {"chunk": d, "page": line_map[question], "source": f}
                chunks_data.append(row)

    connection_args = {"host": "20.115.226.41", "port": "19530"}
    collection_name = "faq"

    schema = create_schema()

    chunk_knowledge = MilvusCollection(connection_args=connection_args,
                                       collection_name=collection_name,
                                       schema=schema,
                                       # recreate_collection=True,
                                       text_field="question")

    chunks_data = pd.DataFrame(chunks_data)

    for ix, row in chunks_data.iterrows():
        print(ix, row)
        chunk_knowledge.add_texts(
            [row["chunk"]],
            metadatas=[
                {
                    "page": int(row["page"]),
                    "source": row["source"]
                }
            ])


if __name__ == '__main__':
    # 1. raw_faq 文件转换为 txt 文件（统一指定格式）
    # raw_convert_txt()

    # # # # 2. 导入embedding database
    insert_embedding_db()

    # refine_answer()
