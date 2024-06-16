import torch
import torch.nn as nn
import torch.optim as optim
import gensim.corpora
import jieba
import os
import re
import math
from torch.utils.data import DataLoader,Dataset
class Configuration:
    def __init__(self):
        self.save_directory = "./models/transformer_model.pt"
        self.ninp = 128  # 词嵌入维度
        self.nhead = 4  # 注意力头数量
        self.nhid = 128  # 隐藏层维度
        self.nlayers = 2  # Transformer 层数
        self.dropout = 0.5  # Dropout 概率
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.transformer = nn.Transformer(ninp, nhead, nlayers, nhid, dropout=dropout)
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()
        self.ninp = ninp

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer(src, src, src_mask, src_mask)
        output = self.decoder(output)
        return output
class TextDataset(Dataset):
    def __init__(self, indices_list, vocab, max_len=50):
        self.vocab = vocab
        self.max_len = max_len
        self.subsequences = []

        # 遍历每个文本的索引列表
        for indices in indices_list:
            # 如果当前文本长度小于max_len，则填充
            if len(indices) < max_len:
                padded_indices = indices + [self.vocab['<pad>']] * (max_len - len(indices))
                self.subsequences.append(padded_indices)
            else:
                # 如果当前文本长度大于或等于max_len，进行切片处理
                self.subsequences.extend([indices[i:i+max_len] for i in range(0, len(indices), max_len) if len(indices[i:i+max_len]) == max_len])

    def __len__(self):
        return len(self.subsequences)

    def __getitem__(self, idx):
        subseq = self.subsequences[idx]
        if len(subseq) != self.max_len:
            raise ValueError('Invalid subsequence length: {}'.format(len(subseq)))
        return torch.tensor(subseq, dtype=torch.long)
# 训练函数
def train(model, data_loader, criterion, optimizer, ntokens, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        src = batch.to(device)
        src_mask = model._generate_square_subsequent_mask(src.size(0)).to(device)
        optimizer.zero_grad()
        output = model(src, src_mask)
        loss = criterion(output.view(-1, ntokens), src.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# 评估函数
def evaluate(model, data_loader, criterion, ntokens, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            src = batch.to(device)
            src_mask = model._generate_square_subsequent_mask(src.size(0)).to(device)
            output = model(src, src_mask)
            loss = criterion(output.view(-1, ntokens), src.view(-1))
            total_loss += loss.item()
    return total_loss / len(data_loader)
# 预测函数
def predict(transformer_model, dictionary, src, device, max_len=50):
    transformer_model.eval()
    with open('./stopwords.txt', 'r', encoding='utf8') as f:
        stop_words = [word.strip() for word in f.readlines()]
    tokenized_texts = []
    for text in [src]:
        # 对每个文本进行分词并过滤停用词
        tokens = [token for token in jieba.lcut(text) if token not in stop_words]
        tokenized_texts.append(tokens)
    # 将输入句子分词并转换为索引
    tokens = tokenized_texts[0]
    # 假设 'dictionary' 已经被正确初始化，并且包含 '<unk>'
    input_indices = [dictionary.token2id.get(token, dictionary.token2id.get('<unk>', -1)) for token in tokens]
    src = torch.tensor(input_indices, dtype=torch.long, device=device).unsqueeze(1)  # (seq_len, batch_size)
    
    # 创建源序列掩码
    src_mask = transformer_model._generate_square_subsequent_mask(src.size(0)).to(device)
    
    # 获取编码器的输出
    with torch.no_grad():
        src = transformer_model.encoder(src) * math.sqrt(transformer_model.ninp)
        src = transformer_model.pos_encoder(src)
        memory = transformer_model.transformer.encoder(src, src_mask)
    
    # 初始化解码器的输入
    input_token = dictionary.token2id.get('<sos>', -1)
    input = torch.tensor([[input_token]], dtype=torch.long, device=device)  # (1, 1)
    
    outputs = []
    
    for _ in range(max_len):
        tgt_mask = transformer_model._generate_square_subsequent_mask(input.size(0)).to(device)
        
        with torch.no_grad():
            input = transformer_model.decoder(input) * math.sqrt(transformer_model.ninp)
            input = transformer_model.pos_encoder(input)
            output = transformer_model.transformer.decoder(input, memory, tgt_mask)
            output = transformer_model.decoder(output)
        
        top1 = output[-1, :, :].argmax(1)
        outputs.append(top1.item())
        
        # 如果预测到结束标志，则停止生成
        if top1.item() == dictionary.token2id.get('<eos>', -1):
            break
        
        # 下一步输入是当前时间步的输出
        input = torch.cat([input, top1.unsqueeze(0)], dim=0)
    
    # 将索引转换回词汇
    output_tokens = [dictionary.idx2word[idx] for idx in outputs if idx in dictionary.idx2word]
    
    return output_tokens
def read_files_from_folder(folder_path):
    total_text = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='ANSI') as file:
                corpus = file.read()
                r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~「」『』（）]+'
                corpus = re.sub(r1, '', corpus)
                corpus = re.sub(r'\n|\u3000|本书来自免费小说下载站|更多更新免费电子书请关注', '', corpus)
                corpus = re.sub(r'[^\u4e00-\u9fff]', '', corpus)
                corpus = corpus.replace(" ", "")
                total_text.append(corpus)
    return total_text

def tokenize(text_list):
    with open('./stopwords.txt', 'r', encoding='utf8') as f:
        stop_words = [word.strip() for word in f.readlines()]
    
    tokenized_texts = []
    for text in text_list:
        # 对每个文本进行分词并过滤停用词
        tokens = [token for token in jieba.lcut(text) if token not in stop_words]
        tokenized_texts.append(tokens)
    
    return tokenized_texts

def build_vocab(tokens, min_freq=2):
    # 创建词典实例
    dictionary = gensim.corpora.Dictionary(tokens)
    
    # 添加特殊标记
    special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
    dictionary.add_documents([[token] for token in special_tokens])  # 添加特殊标记

    # 过滤掉出现频次极低的词汇
    dictionary.filter_extremes(no_below=min_freq, keep_n=None)
    dictionary.compactify()  # 重新分配 id，使其连续

    return dictionary

def text_to_indices(tokens_list, dictionary):
    # 初始化一个空列表用于存储所有文本的索引列表
    indices_list = []

    # 遍历每个文本的词汇列表
    for tokens in tokens_list:
        # 将每个词转换为索引
        indices = [dictionary.token2id[token] if token in dictionary.token2id else dictionary.token2id.get('<unk>') for token in tokens]
        # 将索引列表添加到总的列表中
        indices_list.append(indices)
    
    return indices_list


def create_dataloader(folder_path, batch_size=32, max_len=50):
    texts = read_files_from_folder(folder_path)
    tokenized_texts = tokenize(texts)
    dictionary = build_vocab(tokenized_texts)
    indices_texts = text_to_indices(tokenized_texts, dictionary)
    dataset = TextDataset(indices_texts, dictionary, max_len)
    dictionary.save("./dictionary.bin")
    return dataset, dictionary
def split_dataset(tokenized_data):
    ratio = int(0.9 * len(tokenized_data))
    train_dataset = tokenized_data[:ratio] # 部分作为训练集
    test_dataset = tokenized_data[ratio:] # 部分作为测试集
    return train_dataset, test_dataset
def train_transformer_model(config):
    num_epochs = 1
    batch_size = 32
    dataset, dictionary = create_dataloader('./novels')
    ntokens = len(dictionary)  # 词汇表大小
    train_dataset, test_dataset = split_dataset(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(ntokens, config.ninp, config.nhead, config.nhid, config.nlayers, config.dropout).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dictionary.token2id.get('<pad>'))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer,config.ntokens, device)
        test_loss = evaluate(model, test_loader, criterion, config.ntokens,device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    # torch.save(model.state_dict(), 'transformer_model.pt')
    # 测试预测
    sample_sentence = "青天白日之下，本来万物无怕遁形，但群丐一窝蜂的跟着掌棒龙头和宋青书追出庙门，虽有许多人眼睛一花,"
    print("Predicted text:", predict(model, dictionary, sample_sentence, device))
def main():
    config = Configuration()
    train_transformer_model(config)
if __name__ == "__main__":
    main()