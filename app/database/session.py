#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：beigene-chunk
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/5/11 14:43 
# ====================================
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# 建立数据库连接
engine = create_engine(
    'mysql+pymysql://beigene:Bgn$20230511@beigene-qa.mysql.database.azure.com/beigene-qa',
    pool_size=10, max_overflow=20, pool_timeout=30, pool_recycle=60, pool_pre_ping=True)
# 创建会话工厂
Session = sessionmaker(bind=engine)
# 创建基类
Base = declarative_base()


def init_db():
    # import all modules here that might define models so that
    # they will be registered properly on the metadata.  Otherwise
    # you will have to import them first before calling init_db()
    from . import models
    Base.metadata.create_all(bind=engine)
