#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：beigene-dataprocess
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/4/13 15:07
# ====================================
import json
import sys

import langchain
import os
from typing import List, Tuple
import openai
import pandas as pd
import tiktoken
import tqdm
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import SpacyTextSplitter, LatexTextSplitter, RecursiveCharacterTextSplitter, \
    TokenTextSplitter, MarkdownTextSplitter
from pymilvus import FieldSchema, DataType, CollectionSchema
from langdetect import detect
from store import MilvusCollection
from Cryptodome.Hash import MD4
from Cryptodome.Cipher import AES

openai.api_key = "2e3bc8a0624246b9b5a679f11b5ce5cb"
openai.api_type = "azure"
openai.api_base = "https://apac-openai-service-eus.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
tokenizer = tiktoken.get_encoding("cl100k_base")

TITLE_MAPPING = {
    # v1
    "BeiGene.pdf": "百济神州简介",
    "CN_Corporate_deck.pdf": "百济神州公司介绍",
    "Tisle_NSCLC_NRDL.pdf": "替雷利珠单抗一线NSCLC适应症医保报销沟通材料",
    "Tisle_Spec_20210809.pdf": "替雷利珠单抗注射液说明书",
    "Beigene-hk-2023032901637.pdf": "Beigene-hk ANNUAL RESULTS ANNOUNCEMENT",
    "NRDL_Comm.pdf": "替雷利珠单抗注射液(百泽安)",
    "BeiGene2023-688235_20230228_HGVX.pdf": "百济神州有限公司2023年度报告",
    "NSCLC.pdf": "非鳞癌",
    "POST-ESMO.pdf": "晚期胃癌治究进展与思考",
    "Tisle_HCC.pdf": "百泽安",
    "Tisle_HCC_Comm.pdf": "百泽安HCC患者画像及沟通话术",
    "信达-hk-2023032800723.pdf": "信達生物製藥",
    "君实-hk-2023022700767.pdf": "君实生物医药科技股份有限公司",
}


def length_function(text):
    # return len(tokenizer.encode(text))
    return len(text)


def read_and_split_pdf(pdf_files: List[str], text_splitter) -> List[List[Document]]:
    """
    该函数接受一个pdf文件路径列表，并返回每个pdf文件的页面列表。
    Args:
        pdf_files (List[str]): 包含pdf文件路径的列表。
    Returns:
        List[List[Document]]: 包含每个pdf文件的页面列表的列表。
    """

    pages_list: List[List[Document]] = []
    for pdf_file in pdf_files:
        # 创建一个PyPDFLoader对象，指定要读取的pdf文件路径
        loader = PyPDFLoader(pdf_file)
        # 调用load_and_split方法读取pdf文件并分割成多个页面
        pages: List[Document] = loader.load_and_split(text_splitter=text_splitter)

        pages_list.append(pages)
    return pages_list


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
    chunk = FieldSchema(
        name="chunk",
        dtype=DataType.VARCHAR,
        max_length=6000,
    )
    source = FieldSchema(
        name="source",
        dtype=DataType.VARCHAR,
        max_length=200,
    )
    page = FieldSchema(
        name="page",
        dtype=DataType.INT64
    )
    embedding = FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=1536
    )
    schema = CollectionSchema(
        fields=[id, chunk, embedding, page, source],
        description="document chunk schema",
        auto_id=True
    )
    return schema


def get_pdf_files(root_dir):
    pdf_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files


def extract(pdf_dir, chunk_size=1000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=200,
                                                   length_function=length_function)

    # 获取data目录下所有pdf文件的路径
    # pdf_files: List[str] = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir)]
    pdf_files: List[str] = get_pdf_files(pdf_dir)

    pages: List[List[Document]] = read_and_split_pdf(pdf_files, text_splitter)

    chunks_data = []

    for page in pages:
        for d in page:
            page = d.metadata["page"]
            source = d.metadata["source"]
            page_content = d.page_content
            print("source:", source)
            print("page:", page + 1)

            row = {"chunk": page_content, "page": page, "source": source}
            chunks_data.append(row)
    return chunks_data


def insert(chunks_data, output_id_file):
    connection_args = {"host": "20.115.226.41", "port": "19530"}
    collection_name = "document_chunk_ip"

    schema = create_schema()

    chunk_knowledge = MilvusCollection(connection_args=connection_args,
                                       collection_name=collection_name,
                                       schema=schema,
                                       # recreate_collection=True,
                                       text_field="chunk")

    ids = []
    for ix, row in chunks_data.iterrows():
        print(ix, row)
        id = chunk_knowledge.add_texts(
            [row["chunk"]],
            metadatas=[
                {
                    "page": int(row["page"]),
                    "source": row["source"]
                }
            ])
        ids.append(id[0])

    df = pd.DataFrame(ids)
    df.columns = ["id"]
    df.to_csv(output_id_file)


if __name__ == '__main__':
    # Knowledge 更新

    # 1.  chunk to csv
    # param = {"pdf_dir": "data/pdf", "chunk_size": 1000, "output_dir": "data/output/chunk.csv"}
    # param = {"pdf_dir": "data/RD", "chunk_size": 1000, "output_dir": "data/output/chunk-RD.csv"}
    # param = {"pdf_dir": "data/BGB", "chunk_size": 1000, "output_dir": "data/output/chunk-BGB.csv"}
    param = {"pdf_dir": "data/替雷利珠单抗药学特征", "chunk_size": 1000, "output_dir": "data/output/chunk-替雷利珠单抗药学特征.csv"}

    pdf_dir = param["pdf_dir"]
    chunk_size = param["chunk_size"]
    output = param["output_dir"]

    chunk_data = extract(pdf_dir, chunk_size=chunk_size)
    df = pd.DataFrame(chunk_data)
    df.to_csv(output)

    # # 2. 数据导入 embedding database
    # # param = {"input_file": "data/output/chunk.csv", "output_id_file": "data/milvus_ids/chunk_id.csv"}
    # # param = {"input_file": "data/output/chunk-RD.csv", "output_id_file": "data/milvus_ids/chunk_RD_id.csv"}
    # # param = {"input_file": "data/output/chunk-BGB.csv", "output_id_file": "data/milvus_ids/chunk_BGB_id.csv"}
    # param = {"input_file": "data/output/chunk-替雷利珠单抗药学特征.csv",
    #          "output_id_file": "data/milvus_ids/chunk_替雷利珠单抗药学特征_id.csv"}
    # # insert data in milvus
    #
    # input_file = param["input_file"]
    # output_id_file = param["output_id_file"]
    #
    # chunk_data = pd.read_csv(input_file)
    # insert(chunk_data, output_id_file)
