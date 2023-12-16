#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：beigene-demo
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/4/24 11:36 
# ====================================
import re
from typing import Dict


async def prepare_flask_request(request):
    # If server is behind proxys or balancers use the HTTP_X_FORWARDED fields
    return {
        'https': 'on',  # if request.scheme == 'https' else 'off',
        # 'http_host': request.headers.get("host"),
        'http_host': "bg-chatbot-qa.westus2.cloudapp.azure.com",
        'script_name': request.url.path,
        'get_data': request.query_params,
        # 'lowercase_urlencoding': True,
        'post_data': await request.form()
    }


# 前端引用、后续问题 语言显示
def language_verbose(language: str) -> Dict[str, str]:
    """
    前端引用、后续问题 语言显示
    :param language:
    :return: {"Citations": "Citations:", "Follow-up questions": "Follow-up questions:"}
    """
    verbose_dict = {
        "Chinese": {"Waiting": "深思熟虑中", "Citations": "引用", "Follow-up questions": "后续问题"},
        "English": {"Waiting": "Deep thought", "Citations": "Citations", "Follow-up questions": "Follow-up questions"}
    }
    return verbose_dict.get(language, "English")


# 检索到的Markdown表格文本
def markdown2html(text):
    # 提取表格内容
    table_content = re.findall(r'\|.*\|', text)

    # 解析表格数据
    table_data = [row.strip().split('|') for row in table_content]
    table_data = [row[1:-1] for row in table_data]  # 去掉表格中的首尾空格和竖线

    # 构建HTML表格
    html_table = """<style>
        .styled-table {
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.9em;
        font-family: sans-serif;
        min-width: 400px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    }
    .styled-table thead tr {
        background-color: #009879;
        color: #ffffff;
        text-align: left;
    }
    .styled-table th,
    .styled-table td {
        padding: 12px 15px;
    }
    .styled-table tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    
    .styled-table tbody tr:last-of-type {
        border-bottom: 2px solid #009879;
    }
    .styled-table tbody tr.active-row {
        font-weight: bold;
        color: #009879;
    }
        </style>\n"""
    html_table += '<table class="styled-table">\n'
    for ix, row in enumerate(table_data):
        if ix == 0:
            html_table += '\t<thead>\n'
            html_table += '\t<tr>\n'
            for cell in row:
                html_table += f'\t\t<th>{cell}</th>\n'
            html_table += '\t</tr>\n'
            html_table += '\t</thead>\n'
            html_table += '\t<tbody>\n'
        else:

            html_table += '\t<tr class="active-row">\n'
            for cell in row:
                html_table += f'\t\t<td>{cell}</td>\n'
            html_table += '\t</tr>\n'
    html_table += '\t</tbody>\n'

    html_table += '</table>'

    for n in table_content[:-1]:
        text = text.replace(n, "")
    text = text.replace(table_content[-1], html_table)

    return f"<div>{text}</div>"


def markdown_refine(text):
    # 使用正则表达式查找表格数据
    pattern = r'\|.*?\|\n'
    matches = re.findall(pattern, text, re.DOTALL)

    # 转换表格数据为 Markdown 表格格式
    md_table = ''
    for match in matches:
        md_table += match
        md_table += '\n'

    # 替换原始文本中的表格数据
    text = re.sub(pattern, md_table, text, re.DOTALL)

    # for n in table_content[:-1]:
    #     text = text.replace(n, "")
    # text = text.replace(table_content[-1], html_table)

    return text
