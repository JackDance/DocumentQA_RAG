#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：beigene-chunk
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/5/11 14:45 
# ====================================
from sqlalchemy import Column, Integer, String, Enum, JSON, TEXT, DateTime
from .session import Base
import datetime


class RatingsModel(Base):
    """
    定义Ratings类，用于存储用户的评分信息
    Ratings类继承自Base类，Base类是SQLAlchemy中的基类，用于定义ORM模型
    Ratings类中的每个属性都对应着ratings表中的一个字段
    每个属性都有对应的注释，方便理解
    """
    __tablename__ = 'ratings'

    id = Column(Integer, primary_key=True, comment="自增的主键字段，用于唯一标识每个评分记录")
    user_id = Column(String(512), comment="存储评分的用户ID，可以根据需要进行关联")
    question = Column(String(512), nullable=False, comment="存储与评分相关的问题")
    answer = Column(String(2056), nullable=False, comment="存储与评分相关的回答")
    rating_type = Column(Enum("like", "dislike"), nullable=False, comment="评分类型，使用ENUM数据类型限定为'like'或'dislike'")
    data_points = Column(JSON, nullable=True, comment="用户问题检索到相关的chunk的数据点")
    thoughts = Column(TEXT, nullable=True, comment="检索问题的答案的思路，包括prompt")
    timestamp = Column(DateTime, nullable=False, default=datetime.datetime.now(), comment="评分记录的时间戳，默认为当前时间")

    def __init__(self, user_id=None, question=None, answer=None, rating_type=None, data_points=None, thoughts=None):
        self.user_id = user_id
        self.question = question
        self.answer = answer
        self.rating_type = rating_type
        self.data_points = data_points
        self.thoughts = thoughts

    def __repr__(self):
        return f'<Ratings {self.id!r}>'
