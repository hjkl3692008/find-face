# -*- encoding:utf-8 -*-
"""
    切割训练集为交织，测试
"""
from collections import defaultdict
import itertools
import os

__author__ = 'BBFamily'


def train_val_split(train_path, n_folds=10):
	# 分割不能小于1，如n_folds=10，val占一份，train占9份
    if n_folds <= 1:
        raise ValueError('n_folds must > 1')

    with open(train_path, 'r') as f:
        lines = f.readlines()
        class_dict = defaultdict(list)
        for line in lines:
        	# 找出其分类
            cs = line[line.rfind(' '):]
            # 按分类放入列表
            class_dict[cs].append(line)

    train = list()
    val = list()
    for cs in class_dict:
        cs_len = len(class_dict[cs])
        val_cnt = int(cs_len / n_folds)
        val.append(class_dict[cs][:val_cnt])
        train.append(class_dict[cs][val_cnt:])
    # 扁平化，例[[1,2],[3,4]]->[1,2,3,4]
    val = list(itertools.chain.from_iterable(val))
    train = list(itertools.chain.from_iterable(train))
    test = [t.split(' ')[0] + '\n' for t in val]

    fn = os.path.dirname(train_path) + '/train_split.txt'
    with open(fn, 'w') as f:
        f.writelines(train)
    fn = os.path.dirname(train_path) + '/val_split.txt'
    with open(fn, 'w') as f:
        f.writelines(val)
    fn = os.path.dirname(train_path) + '/test_split.txt'
    with open(fn, 'w') as f:
        f.writelines(test)
