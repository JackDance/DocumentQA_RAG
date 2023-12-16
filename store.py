# !/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：beigene-demo
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/4/12 9:49
# ====================================
import os

import openai
import pandas as pd
import tqdm
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from pymilvus import CollectionSchema, FieldSchema, DataType, connections

from typing import Text, Dict, Optional, Iterable, List, Any
from pymilvus import utility
from pymilvus import Collection

DEFAULT_CONNECTION_ARGS = {"host": "localhost", "port": "19530"}
DEFAULT_CONNECTION_NAME = "faq"

os.environ["OPENAI_API_KEY"] = "2e3bc8a0624246b9b5a679f11b5ce5cb"
openai.api_key = "2e3bc8a0624246b9b5a679f11b5ce5cb"
openai.api_type = "azure"
# openai.api_base = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
openai.api_base = "https://apac-openai-service-eus.openai.azure.com/"
openai.api_version = "2023-03-15-preview"


class MilvusExtension:

    def create_collection(self, collection_name, schema):
        """
        Creates a collection with the given name and schema.

        :param collection_name: The name of the collection.
        :type collection_name: str
        """
        index_params = self.index()

        self.collection = Collection(
            name=collection_name,
            schema=schema,
            using='default',
            shards_num=2
        )

        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

        self.collection.flush()

    def index(self):
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }

        return index_params

    def delete_collection(self, collection_name):
        """
        Deletes the collection with the given name.

        :param collection_name: The name of the collection.
        :type collection_name: str
        """
        utility.drop_collection(collection_name)

    def recreate_collection(self, collection_name, schema):
        """
        Deletes the collection with the given name and recreates it.

        :param collection_name: The name of the collection.
        :type collection_name: str


        example:
            >>> connection_args = {"host": "20.115.226.41", "port": "19530"}
            >>> collection_name = "faq"
            >>> faq = MilvusFAQ(connection_args, collection_name)
            >>> faq.recreate_collection(collection_name)

        """
        self.delete_collection(collection_name)
        self.create_collection(collection_name, schema)


class MilvusCollection(Milvus, MilvusExtension):
    """
    The MilvusFAQ class is used to interact with a Milvus collection containing frequently asked questions (FAQs).
    """

    def __init__(
            self,
            connection_args: Optional[Dict] = None,
            collection_name: Optional[Text] = None,
            schema: CollectionSchema = None,
            text_field: Text = None,
            recreate_collection: bool = False
    ) -> None:
        """
        Initializes the MilvusFAQ class with the given connection arguments and collection name.

        :param connection_args: A dictionary containing the connection arguments.
        :type connection_args: dict
        :param collection_name: The name of the collection.
        :type collection_name: str
        """

        connection_args = connection_args or DEFAULT_CONNECTION_ARGS
        collection_name = collection_name or DEFAULT_CONNECTION_NAME

        connections.connect(**connection_args)

        if recreate_collection:
            self.recreate_collection(collection_name, schema)
        self.collection_name = collection_name

        embedding_function = OpenAIEmbeddings()

        super().__init__(embedding_function, connection_args, collection_name, text_field)


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


def handle_internal_faq():
    """处理内部faq文件"""

    # 内部文件列表
    internal_file_list = [
        "凯洛斯问答机器人.xlsx",
        "安加维百科.xlsx",
        "百汇泽百科全书.xlsx"
    ]

    questions = []
    answers = []
    with open("data/raw_faq/智能免疫助手QnA.txt", "r", encoding="utf8") as f:
        data = f.readlines()
        simlarity_question = False
        for d in data:
            if d.startswith("相似问题："):
                questions.append(d.strip().replace("相似问题：", ""))
                simlarity_question = True
            elif d.startswith("问题："):
                questions.append(d.strip().replace("问题：", ""))
            elif d.startswith("回答："):
                a = d.strip().replace("回答：", "").replace('\\n', '\n')
                answers.append(a)
                if simlarity_question == True:
                    answers.append(a)
                    simlarity_question = False

            else:
                pass

    excel_files: List[str] = [os.path.join("data/raw_faq", f) for f in os.listdir("data/raw_faq") if
                              f in internal_file_list]
    for e in excel_files:
        k_df = pd.read_excel(e)
        question_column = "标准问题(80字以内，不能为空)"
        answer_column = "答案(2000字以内，不能为空)"
        sim_question_columns = [i for i in k_df.columns if i.startswith("相似问法")]
        k_df.fillna("0", inplace=True)
        for ix, row in k_df.iterrows():
            q = [row[question_column]] + row[sim_question_columns].tolist()
            q = [i for i in q if i != "0"]
            a = [row[answer_column]] * len(q)

            questions.extend(q)
            answers.extend(a)

    return questions, answers


def handle_external_faq():
    """处理外部faq文件"""
    # external_file_list = ["千方百济-智能客服配置.xlsx"]
    # questions, answers = [], []
    #
    # excel_files: List[str] = [os.path.join("data/raw_faq", f) for f in os.listdir("data/raw_faq") if f in external_file_list]
    # for e in excel_files:
    #     k_df = pd.read_excel(e, engine="openpyxl")
    #     question_column = "基础问题"
    #     answer_column = "微信公众号推文链接"
    #     sim_question_columns = [i for i in k_df.columns if i.startswith("关联问题")]
    #     k_df.fillna("0", inplace=True)
    #     for ix, row in k_df.iterrows():
    #         q = [row[question_column]] + row[sim_question_columns].tolist()
    #         q = [i for i in q if i != "0"]
    #         a = [f"我找到了一篇公众号推文,了解详情：[链接]({row[answer_column].strip()})"] * len(q)
    #
    #         questions.extend(q)
    #         answers.extend(a)

    df = pd.read_csv("data/raw_faq/千方百济.csv")

    questions = df["question"].tolist()
    answers = df["answer"].tolist()

    return questions, answers


def handle_append_faq():
    questions = []
    answers = []
    with open("data/raw_faq/替雷利珠单抗蛋白质序列.txt", "r", encoding="utf8") as f:
        data = f.readlines()
        simlarity_question = False
        for d in data:
            if d.startswith("相似问题："):
                questions.append(d.strip().replace("相似问题：", ""))
                simlarity_question = True
            elif d.startswith("问题："):
                questions.append(d.strip().replace("问题：", ""))
            elif d.startswith("回答："):
                a = d.strip().replace("回答：", "").replace('\\n', '\n')
                answers.append(a)
                if simlarity_question == True:
                    answers.append(a)
                    simlarity_question = False

            else:
                pass
    return questions, answers


if __name__ == '__main__':

    # Faq 数据更新

    # milvus 链接参数
    connection_args = {"host": "20.115.226.41", "port": "19530"}

    # 内部 faq 集合名称
    collection_name = "faq"

    schema = create_schema()

    faq = MilvusCollection(connection_args=connection_args,
                           collection_name=collection_name,
                           schema=schema,
                           text_field="question",
                           # recreate_collection=True
                           )

    # 处理内部faq
    # questions_a, answers_a = handle_internal_faq()
    #
    # # # 处理外部faq
    # questions_b, answers_b = handle_external_faq()
    #
    # questions = questions_a + questions_b
    # answers = answers_a + answers_b

    questions, answers = handle_append_faq()



    for q, a in tqdm.tqdm(zip(questions, answers), total=len(questions)):
        # print(q)
        # print(a)
        # print("\n\n")
        faq.add_texts([q.strip()], metadatas=[{"answer": a.strip()}])
