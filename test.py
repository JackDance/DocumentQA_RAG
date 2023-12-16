#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：beigene-async
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/8/30 13:40 
# ====================================


import os


def get_pdf_files(root_dir):
    pdf_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files


data_dir = "data"
pdf_files = get_pdf_files(data_dir)

for pdf_file in pdf_files:
    print(pdf_file)
