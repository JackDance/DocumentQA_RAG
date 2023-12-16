#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：beigene-async
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/6/30 17:37 
# ====================================
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel


class Rating(BaseModel):
    question: str
    answer: str
    rating_type: str
    data_points: List[str]
    thoughts: str


class RatingCancel(BaseModel):
    id: int
