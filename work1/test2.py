import jieba
import math

class Novels:
    def __init__(self, name):
        self.data = None
        self.name = name
        # 单个字
        self.word = []  # 单个字列表
        self.word_len = 0  # 单个字总字数
        # 词
        self.split_word = []  # 单个词列表
        self.split_word_len = 0  # 单个词总数
        with open("cn_stopwords.txt", "r", encoding='utf-8') as f:
            self.stop_word = f.read().split('\n')
            f.close()

    def read_file(self, filename=""):
        # 如果未指定名称，则默认为类名
        if filename == "":
            filename = self.name
        target = 'jyxstxtqj_downcc.com/' + filename + ".txt"
        with open(target, "r", encoding='gbk', errors='ignore') as f:
            self.data = f.read()
            self.data = self.data.replace(
                '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')
            self.data = self.data.replace('新语丝电子文库(www.xys.org)','')
            self.data = self.data.replace('新语丝电子文库','')
            self.data = self.data.replace('Last Updated: Saturday, November 16, 1996','')
            self.data = self.data.replace(u'\u3000',u'').replace('\n','').replace('\r','').replace(" ","")
            self.data = self.data.replace('[','').replace(']','')
            f.close()
        # 分词
        for words in jieba.cut(self.data):
            if (words not in self.stop_word) and (not words.isspace()):
                self.split_word.append(words)
                self.split_word_len += 1
        # 统计字数
        for word in self.data:
            if (word not in self.stop_word) and (not word.isspace()):
                self.word.append(word)
                self.word_len += 1

    def write_file(self):
        # 将文件内容写入总文件
        target = "jyxstxtqj_downcc.com/data.txt"
        with open(target, "a") as f:
            f.write(self.data)
            f.close()

    def get_unigram_tf(self, word):
        # 得到单个词的词频表
        unigram_tf = {}
        for w in word:
            unigram_tf[w] = unigram_tf.get(w, 0) + 1
        return unigram_tf

    def get_bigram_tf(self, word):
        # 得到二元词的词频表
        bigram_tf = {}
        for i in range(len(word) - 1):
            bigram_tf[(word[i], word[i + 1])] = bigram_tf.get(
                (word[i], word[i + 1]), 0) + 1
        return bigram_tf

    def get_trigram_tf(self, word):
        # 得到三元词的词频表
        trigram_tf = {}
        for i in range(len(word) - 2):
            trigram_tf[(word[i], word[i + 1], word[i + 2])] = trigram_tf.get(
                (word[i], word[i + 1], word[i + 2]), 0) + 1
        return trigram_tf

    def calc_entropy_unigram(self, word):
        # 计算一元模型的信息熵
        word_tf = self.get_unigram_tf(word)
        word_len = sum([item[1] for item in word_tf.items()])
        entropy = sum(
            [-(word[1] / word_len) * math.log(word[1] / word_len, 2) for word in
             word_tf.items()])
        return entropy

    def calc_entropy_bigram(self, word):
        # 计算二元模型的信息熵
        # 计算二元模型总词频
        word_tf = self.get_bigram_tf(word)
        last_word_tf = self.get_unigram_tf(word)
        bigram_len = sum([item[1] for item in word_tf.items()])
        entropy = []
        for bigram in word_tf.items():
            p_xy = bigram[1] / bigram_len  # 联合概率p(xy)
            p_x_y = bigram[1] / last_word_tf[bigram[0][0]]  # 条件概率p(x|y)
            entropy.append(-p_xy * math.log(p_x_y, 2))
        entropy = sum(entropy)
        return entropy

    def calc_entropy_trigram(self, word):
        # 计算三元模型的信息熵
        # 计算三元模型总词频
        word_tf = self.get_trigram_tf(word)
        last_word_tf = self.get_bigram_tf(word)
        trigram_len = sum([item[1] for item in word_tf.items()])
        entropy = []
        for trigram in word_tf.items():
            p_xy = trigram[1] / trigram_len  # 联合概率p(xy)
            p_x_y = trigram[1] / last_word_tf[(trigram[0][0], trigram[0][1])]  # 条件概率p(x|y)
            entropy.append(-p_xy * math.log(p_x_y, 2))
        entropy = sum(entropy)
        return entropy

if __name__ == "__main__":
    data_set_list = []
    # 每次运行程序将总内容文件清空
    with open("jyxstxtqj_downcc.com/data.txt", "w") as f:
        f.close()
    with open("log.txt", "w") as f:
        f.close()
    # 读取小说名字
    with open("jyxstxtqj_downcc.com/inf.txt", "r") as f:
        txt_list = f.read().split(',')
        i = 0
        for name in txt_list:
            locals()[f'set{i}'] = Novels(name)
            data_set_list.append(locals()[f'set{i}'])
            i += 1
        f.close()
    # 分别针对每本小说进行操作
    word_unigram_entropy, word_bigram_entropy, word_trigram_entropy, words_unigram_entropy, words_bigram_entropy, words_trigram_entropy = [], [], [], [], [], []
    for set in data_set_list:
        set.read_file()
        set.write_file()
        # 字为单位
        word_unigram_entropy.append(set.calc_entropy_unigram(set.word))
        word_bigram_entropy.append(set.calc_entropy_bigram(set.word))
        word_trigram_entropy.append(set.calc_entropy_trigram(set.word))
        # 词为单位
        words_unigram_entropy.append(set.calc_entropy_unigram(set.split_word))
        words_bigram_entropy.append(set.calc_entropy_bigram(set.split_word))
        words_trigram_entropy.append(set.calc_entropy_trigram(set.split_word))
        with open("log.txt", "a") as f:
            f.write("{:<10} 字数：{:10d} 词数：{:10d} 信息熵：{:.4f}、{:.4f}、{:.4f}、{:.4f}、{:.4f}、{:.4f}\n".format(set.name,
                                                                                                               set.word_len,
                                                                                                               set.split_word_len,
                                                                                                               word_unigram_entropy[
                                                                                                                   -1],
                                                                                                               word_bigram_entropy[
                                                                                                                   -1],
                                                                                                               word_trigram_entropy[
                                                                                                                   -1],
                                                                                                               words_unigram_entropy[
                                                                                                                   -1],
                                                                                                               words_bigram_entropy[
                                                                                                                   -1],
                                                                                                               words_trigram_entropy[
                                                                                                                   -1]))
            f.close()
    # 对所有小说进行操作
    set_total = Novels("total")
    set_total.read_file("data")
    word_unigram_entropy.append(set_total.calc_entropy_unigram(set_total.word))
    word_bigram_entropy.append(set_total.calc_entropy_bigram(set_total.word))
    word_trigram_entropy.append(set_total.calc_entropy_trigram(set_total.word))

    words_unigram_entropy.append(set_total.calc_entropy_unigram(set_total.split_word))
    words_bigram_entropy.append(set_total.calc_entropy_bigram(set_total.split_word))
    words_trigram_entropy.append(set_total.calc_entropy_trigram(set_total.split_word))

    with open("log.txt", "a") as f:
        f.write(
            "{:<10} 字数：{:10d} 词数：{:10d} 信息熵：{:.4f}、{:.4f}、{:.4f}、{:.4f}、{:.4f}、{:.4f}\n".format(set_total.name,
                                                                                                       set_total.word_len,
                                                                                                       set_total.split_word_len,
                                                                                                       word_unigram_entropy[
                                                                                                           -1],
                                                                                                       word_bigram_entropy[
                                                                                                           -1],
                                                                                                       word_trigram_entropy[
                                                                                                           -1],
                                                                                                       words_unigram_entropy[
                                                                                                           -1],
                                                                                                       words_bigram_entropy[
                                                                                                           -1],
                                                                                                       words_trigram_entropy[
                                                                                                           -1]))
        f.close()
    
