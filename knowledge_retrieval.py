#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     knowledge_retrieval.py
   @Author:        Luyao.zhang
   @Date:          2023/12/29
   @Description:
-------------------------------------------------
"""
import os
from dotenv import load_dotenv
from typing import List
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain.vectorstores import Milvus
from pymilvus import connections, Collection
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv(".env")

# ------------ OpenAI Configuration -----------
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY1")
openai_api_key = os.environ.get("OPENAI_API_KEY1")
emb_openai_api_base = os.environ.get("EMB_OPENAI_API_BASE")
chat_openai_api_base = os.environ.get("CHAT_OPENAI_API_BASE")
os.environ["openai_api_key"] = openai_api_key
os.environ["openai_api_base"] = chat_openai_api_base
# ------------ Milvus -----------
connection_args = {"host": os.environ.get("MILVUS_HOST"), "port": os.environ.get("MILVUS_PORT")}


class RAG_pipeline(object):
    """
    构建RAG Pipeline
    """

    def __init__(self, milvus_connection_args, embeddings_model, query: str):
        """
        Args:
            doc_folder: txt本地文件夹路径
            milvus_connection_args: milvus连接信息
            embeddings_model: text嵌入模型
            query: 用户输入的请求
        """
        self.milvus_connection_args = milvus_connection_args
        self.embeddings_model = embeddings_model
        self.query = query

    def child_chunk_retriever(
            self,
            child_collection_name: str = "child_chunk"):
        """
        对child_chunk设置检索器retriever
        Returns:

        """
        # 构建用于索引child chunk的向量数据库
        vectorstore = Milvus(
            connection_args=self.milvus_connection_args,
            collection_name=child_collection_name,
            embedding_function=self.embeddings_model,
        )
        # The storage layer for the parent documents
        store = InMemoryByteStore()
        id_key = "doc_id"
        # The retriever (empty to start)
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=store,
            id_key=id_key,
        )

        # Vectorstore alone retrieves the child chunks
        # retriever.search_type = SearchType.mmr
        child_chunk_res = retriever.vectorstore.similarity_search(
            query=self.query, k=3)
        # return child_chunk_res

        # 只返回doc_id组成的列表
        res_id_list = []
        for single_res in child_chunk_res:
            res_id_list.append(single_res.metadata["doc_id"])
        return res_id_list

        # Retriever returns larger chunks
        # large_chunk_res = retriever.get_relevant_documents(query=query)
        # print(large_chunk_res)

    def summary_retriever(
            self,
            summary_collection_name: str = "summary"):
        """
        对summary_chunk设置检索器retriever
        Returns:
        """
        # 构建用于索引summary的向量数据库
        vectorstore = Milvus(
            connection_args=self.milvus_connection_args,
            collection_name=summary_collection_name,
            embedding_function=self.embeddings_model,
        )
        # The storage layer for the parent documents
        store = InMemoryByteStore()
        id_key = "doc_id"
        # The retriever (empty to start)
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=store,
            id_key=id_key,
        )

        # Vectorstore alone retrieves the summary chunks
        # retriever.search_type = SearchType.mmr
        summary_chunk_res = retriever.vectorstore.similarity_search(
            query=self.query, k=3)
        # return summary_chunk_res

        # 只返回doc_id组成的列表
        res_id_list = []
        for single_res in summary_chunk_res:
            res_id_list.append(single_res.metadata["doc_id"])
        return res_id_list

    def hypothetical_retriever(
        self,
        hypothetical_query_collection="hypothetical_query"
    ):
        """
        对hypothetical_query设置检索器retriever
        Returns:

        """
        pass

    def get_parent_document(
            self,
            doc_id_list: List[str],
            parent_chunk: str = "parent_chunk"):
        connections.connect(**self.milvus_connection_args)
        collection = Collection(name=parent_chunk)
        parent_chunk_list = []
        for doc_id in doc_id_list:
            # 向量查询
            retrieved_res = collection.query(
                expr=f"doc_id in ['{doc_id}']",
                offset=0,
                limit=10,
                output_fields=["text"],
                consistency_level="Strong"
            )
            parent_chunk_list.append(retrieved_res[0]["text"])

        return parent_chunk_list

    def build_prompt_get_answer(self, docs: List[str], stream=True):
        """
        构建prompt并返回LLM的答案
        """
        template = """
        你是一个文档问答机器人，请仅仅根据下面指定文档列表中的多个文档来回答提出的问题，不能依赖自己的任何先验知识，如果在指定的文档中没有找到问题的答案，
        请回答:'抱歉，本地知识库中暂无该问题相关的信息。'
        {context}
        问题：{question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(temperature=0.1,
                           openai_api_key=openai_api_key,
                           openai_api_base=chat_openai_api_base,
                           model_name="qwen-14b-chat"
                           )
        if stream:
            llm_chain = prompt | model
            answer = llm_chain.stream(
                {"question": self.query, "context": docs})
            for ret in answer:
                yield ret.content
        else:
            chain = LLMChain(llm=model, prompt=prompt)
            answer = chain.run({"question": self.query, "context": docs})
            return answer

    def build_rag(self, stream=True):
        """
        构建RAG Pipeline
        Returns:
        """
        child_list = self.child_chunk_retriever()
        summary_list = self.summary_retriever()
        # 取交集
        intersection_list = list(set(child_list) & set(summary_list))
        parent_chunk_list = self.get_parent_document(
            intersection_list if intersection_list else child_list)
        if stream:
            stream_res = self.build_prompt_get_answer(
                parent_chunk_list, stream=stream)
            for res in stream_res:
                print(res, end="", flush=True)
        else:
            answer = self.build_prompt_get_answer(
                parent_chunk_list, stream=stream)
            return answer


def main():
    while True:
        input_prompt = "\n请输入内容（输入 'exit' 退出程序）: "
        query = input(input_prompt)
        if query.lower() == "exit":
            print("程序退出。")
            break
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, openai_api_base=emb_openai_api_base)
        rag = RAG_pipeline(milvus_connection_args=connection_args, embeddings_model=embeddings, query=query)
        rag.build_rag()



if __name__ == '__main__':
    main()