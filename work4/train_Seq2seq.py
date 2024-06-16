import torch
import torch.nn as nn
import torch.optim as optim
import gensim.corpora
import jieba
import opencc
import random
import tqdm
import os
import numpy as np
import re
class Configuration:
    def __init__(self):
        self.dictionary = gensim.corpora.Dictionary.load("./result/dictionary.bin")
        self.save_directory = "./models/seq2seq_model.pt"
        self.vocab_size = len(self.dictionary)
        self.bos_id = 0
        self.eos_id = 1
        self.pad_id = 2
        self.max_tokens = 128
        self.embedding_dim = 128
        self.hidden_dim = 128
        self.num_layers = 2
        self.learning_rate = 0.001
        self.num_epochs = 50
def is_chinese_token(s):
    if len(s) == 1 and s[0] in ["“", "”", "：", "，", "。", "？", "！", "（", "）", "…", "、", "；"]:
        return True
    for c in s:
        if not "\u4e00" <= c <= "\u9fff":
            return False
    return True
def load_tokenize(config):
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
        sentences_tokenized = [[config.dictionary.token2id[token] for token in jieba.cut(sentence, cut_all=False) if is_chinese_token(token)] for sentence in cleaned_parts] # 句子转token id
        corpus_tokenized.extend(sentences_tokenized) # 加入到总数据集
    return corpus_tokenized
def split_dataset(tokenized_data):
    ratio = int(0.9 * len(tokenized_data))
    random.shuffle(tokenized_data)
    train_dataset = tokenized_data[:ratio] # 部分作为训练集
    test_dataset = tokenized_data[ratio:] # 部分作为测试集
    return train_dataset, test_dataset
class Seq2SeqModel(nn.Module):
    def __init__(self, args):
        super(Seq2SeqModel, self).__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.lstm = nn.LSTM(args.embedding_dim, args.hidden_dim, args.num_layers)
        self.linear = nn.Linear(args.hidden_dim, args.vocab_size)
    def forward(self, x, hx):
        x = self.embedding(x)
        out, hx = self.lstm(x, hx)
        out = self.linear(out)
        return out, hx
    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.05, 0.05)
    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.05, 0.05)
    def predict(self, input_tensor,config):
        self.eval()
        with torch.no_grad():
            for src in input_tensor:
                input_token_ids = [config.bos_id] + src[: len(src) // 2 if len(src) // 2 != 0 else 1] # 取前一半作为输入，并加入bos
                i = 0
                while i < len(input_token_ids): # 前一半使用真实标签，得到隐藏状态和最后的输出
                    if i == 0:
                        hx = None
                    input_tensor = torch.Tensor([input_token_ids[i]]).long().cuda()
                    out, hx = self(input_tensor, hx)
                    i += 1
                out = torch.argmax(out).unsqueeze(0)
                while True: # 后面完全使用当前输出和隐藏状态来推理
                    out, hx = self(out, hx)
                    out_id = out.clone().detach().cpu().numpy()
                    out_id = np.argmax(out_id)
                    input_token_ids.append(out_id)
                    if out_id == config.eos_id or len(input_token_ids) >= 128:
                        break
                    out = torch.argmax(out).unsqueeze(0)
                print("提示词：" + "".join([config.dictionary[token_id] for token_id in src[: len(src) // 2 if len(src) // 2 != 0 else 1]]))
                print("真实标签：" + "".join([config.dictionary[token_id] for token_id in src]))
                print("提示词+生成：" + "".join([config.dictionary[token_id] for token_id in input_token_ids]))
                print("\n")
def train(config):
    # Load and preprocess the corpus
    corpus_tokenized = load_tokenize(config)
    train_dataset, test_dataset = split_dataset(corpus_tokenized)
    
    # Initialize the model, optimizer, and loss function
    model = Seq2SeqModel(config)
    if os.path.exists(config.save_directory): # 加载模型参数
        model.load_state_dict(torch.load(config.save_directory, map_location=torch.device('cpu')), strict=True)
    else:
        model.init_weights() # 初始化权重
    model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda() # 损失函数
    tq = tqdm.tqdm(total=config.num_epochs * len(train_dataset))
    global_step = 1 # 当前步数
    for epoch in range(config.num_epochs):
        total_loss = 0
        select = list(range(len(train_dataset)))
        random.shuffle(select)
        # Training loop
        for s in select:
            sentence_tokenized = train_dataset[s] # 选择第s个句子
            seq = [config.bos_id] + sentence_tokenized + [config.eos_id] # 加入开始和结束标志
            total_length = len(seq) # 总长度
            for i in range(total_length - 1):
                if i == 0: # 第一个输入不包含隐藏状态
                    hx = None
                model.train()
                input_seq = torch.Tensor([seq[i]]).long().cuda() # 当前位置为输入
                label_seq = torch.Tensor([seq[i + 1]]).long().cuda() # 下一位置为标签
                optimizer.zero_grad()
                rand_num = random.randint(1, 100)
                if 1 <= rand_num <= 75 or i == 0: # 一定几率使用真值训练
                    out, hx = model(input_seq, hx)
                else: # 一定几率使用上次输出训练
                    out = torch.argmax(out).unsqueeze(0)
                    out, hx = model(out, hx)
                loss = criterion(out, label_seq) # 计算损失值
                loss.backward()
                optimizer.step()
                hx0 = hx[0].detach()
                hx1 = hx[1].detach()
                hx = (hx0, hx1)
                tq.set_postfix(loss=loss.item(), lr=config.learning_rate)
                tq.update(1)
                global_step += 1
                if global_step % 1000 == 0: # 每1000步评估模型
                    model.predict(random.choices(test_dataset, k=10), config)
                if global_step % 10000 == 0: # 每10000步保存模型
                    torch.save(model.state_dict(), config.save_directory)
    tq.close()
def main():
    config = Configuration()
    train(config)
if __name__ == "__main__":
    main()