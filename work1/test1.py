import jieba
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
# 加载停用词表
stopwords_file = 'cn_stopwords.txt'
stopwords = set()
with open(stopwords_file, 'r', encoding='utf-8') as f:
    for line in f:
        stopwords.add(line.strip())
    f.close()

# 从inf.txt中读取小说文件名
novel_file = 'D:/课程/研一下NLP/work1/jyxstxtqj_downcc.com/inf.txt'
with open(novel_file,'r',encoding='ANSI') as f:
    novel_files = f.read().split(',')
    f.close()

# 读取文本数据
novels = []

for novel_file in novel_files:
    novel_file_name = 'D:/课程/研一下NLP/work1/jyxstxtqj_downcc.com/'+novel_file+'.txt'
    with open(novel_file_name,'r',encoding='ANSI') as f:
        novel_text = f.read()
        novel_text = novel_text.replace(
                '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')
        novel_text = novel_text.replace('新语丝电子文库(www.xys.org)','')
        novel_text = novel_text.replace('新语丝电子文库','')
        novel_text = novel_text.replace('Last Updated: Saturday, November 16, 1996','')
        novel_text = novel_text.replace(u'\u3000',u'').replace('\n','').replace('\r','').replace(" ","")
        novel_text = novel_text.replace('[','').replace(']','')
        novels.append(novel_text)
        f.close()
    
i=0
# 创建4x4的子图布局
fig, axs = plt.subplots(4, 4, figsize=(16, 16))

# 分词和去除停用词
words = []

for novel in novels:
    seg_list = jieba.cut(novel)
    filtered_words = [word for word in seg_list if word not in stopwords]
    words.extend(filtered_words)
    # 统计词频
    word_freq = {}
    for word in filtered_words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    # 根据词频进行排序
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    # 提取词频和词序排名
    word_ranks = list(range(1, len(sorted_word_freq) + 1))
    word_frequencies = [freq for _, freq in sorted_word_freq]
    # 绘制词频与词序排名的关系图
    ax = axs[i // 4, i % 4]
    ax.plot(word_ranks, word_frequencies, linewidth=2, color='blue')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Word Rank')
    ax.set_ylabel('Word Frequency')
    ax.set_title(novel_files[i])
    i+=1
plt.figure(1)
plt.plot()#画在图1上
# print(words)
# 统计词频并排序
word_freq = {}
for word in words:
    if word in word_freq:
        word_freq[word] += 1
    else:
        word_freq[word] = 1

sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

# 绘制词频与词序排名的关系图
word_ranks = list(range(1, len(sorted_word_freq) + 1))
word_freqs = [freq for _, freq in sorted_word_freq]
plt.figure(2)
plt.plot(word_ranks, word_freqs)
plt.xlabel('Word Rank')
plt.ylabel('Word Frequency')
plt.xscale('log')
plt.yscale('log')
plt.title('Zipf\'s Law')
plt.show()