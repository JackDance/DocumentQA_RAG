#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：beigene-chunk
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/5/29 16:48 
# ====================================
import os

import pandas as pd
from pymilvus import connections
from pymilvus import Collection

# Milvus 连接参数
connection_args = {"host": "20.115.226.41", "port": "19530"}
collection_name = "document_chunk"
# 删选条件
delete_source_list = ["凯洛斯问答机器人.txt", "千方百济.txt", "安加维百科.txt", "智能免疫助手QnA.txt", "百汇泽百科全书.txt"]

connections.connect(**connection_args)

collection = Collection(collection_name)


def delete_entity(id):
    """
    删除 milvus 中实体对象
    :param id:
    :return:
    """
    expr = f"id in [{id}]"
    res = collection.delete(expr)
    print(res)


for i in range(0, 200):
    offset = i
    limit = i + 5000
    res = collection.query(
        expr="id >= 0",
        offset=offset,
        limit=limit,
        output_fields=["source", "page", "chunk"])

    for i in res:
        source = i["source"]
        page = i["page"]
        id = i["id"]
        print(source, page)

        source = os.path.basename(source)

        if source in delete_source_list:
            delete_entity(id)
