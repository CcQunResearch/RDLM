# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 18:59
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : utils.py
# @Software: PyCharm
# @Note    :

def write_log(log, str):
    log.write(f'{str}\n')
    log.flush()
