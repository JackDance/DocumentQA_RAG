#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：beigene-demo
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：04/11/2023 9:08
# ====================================
import abc


class Approach(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def run(self, history: list[dict], overrides: dict, model: str) -> any:
        raise NotImplementedError
