#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：beigene-async
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/6/30 14:09 
# ====================================
from typing import List, Dict

from pydantic import BaseModel


class Override(BaseModel):
    semantic_ranker: bool
    semantic_captions: bool
    suggest_followup_questions: bool
    top: int


class Chat(BaseModel):
    history: List
    approach: str
    overrides: Override
    model: str = "chat"
