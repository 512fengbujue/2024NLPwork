import jieba
import opencc
import re
import numpy as np
import gensim.corpora
import gensim.models
def is_chinese_token(s):
    if len(s) == 1 and s[0] in ["“", "”", "：", "，", "。", "？", "！", "（", "）", "…", "、", "；"]:
        return True
    for c in s:
        if not "\u4e00" <= c <= "\u9fff":
            return False
    return True
cc = opencc.OpenCC("t2s") # 初始化opencc，繁体转简体
corpus_path = "./novels" # 语料库路径
title_list = open("{}/inf.txt".format(corpus_path), "r", encoding="gb18030").readline().split(",")
corpus_tokenized = list()
useless = [
    "\n",
    "　",
    " ",
    "本书来自www.cr173.com免费txt小说下载站",
    "更多更新免费电子书请关注www.cr173.com",
]
replace_context = [
    ["「", "“"],
    ["」", "”"],
    ["『", "“"],
    ["』", "”"],
    [":", "："],
    [",", "，"],
    [".", "。"],
    ["?", "？"],
    ["!", "！"],
    ["(", "（"],
    [")", "）"],

]
for title in title_list:
    book = open("{}/{}.txt".format(corpus_path, title), "r", encoding="gb18030").read() # 按对应编码格式打开文件并读取内容
    for word in useless:
        book = book.replace(word, "")
    for replace in replace_context:
        book = book.replace(replace[0], replace[1])
    # 统一省略号
    book = re.sub(r"…+", "…", book)
    book = cc.convert(book) # 繁体转简体
    parts = re.split('([。！？…])', book)
    # 过滤掉空字符串，并确保每个有效部分末尾保留句号、叹号或问号
    cleaned_parts = [part + punctuation for part, punctuation in zip(parts[::2], parts[1::2])]
    # 如果原文以句号、叹号、问号或省略号结尾，最后一个元素会是多余的标点，需要去掉
    if cleaned_parts and cleaned_parts[-1] in '。！？…':
        cleaned_parts.pop()
    # 对每一个句子进行分词，分词结果保存在list[list[str]]，
    # 外层list保存句子，内层list为每个句子分词后的结果
    # 要求词包含中文字符
    sentences_tokenized = [[word for word in jieba.cut(sentence, cut_all=False) if is_chinese_token(word)] for sentence in cleaned_parts]
    sentences_tokenized = [s for s in sentences_tokenized if len(s) != 0] # 去除空句子（可能整句都在停词表中）
    corpus_tokenized.extend(sentences_tokenized) # 加入到总的list中
dictionary = gensim.corpora.Dictionary([["<bos>", "<eos>", "<pad>"]]) # 创建字典，加入<bos><eos><pad>
dictionary.add_documents(corpus_tokenized) # 将预料库加入字典
dictionary.save("./result/dictionary.bin") # 保存字典