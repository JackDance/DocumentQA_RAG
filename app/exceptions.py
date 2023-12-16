#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：beigene-async
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/8/2 14:18 
# ====================================


class OpenAITimeoutException(Exception):
    def __init__(self, name: str):
        self.name = name


class TranslatorException(Exception):
    def __init__(self, name: str):
        self.name = name
