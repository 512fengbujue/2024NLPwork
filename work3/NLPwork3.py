import numpy as np
import re
import os
import jieba
from gensim.models import Word2Vec,word2vec
import pickle
import multiprocessing
import math
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import tqdm
def preprocess_text(novel_name):
    with open(novel_name, 'r', encoding='ANSI') as f:
        con = f.read()
        ad = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库','Last Updated: Saturday, November 16, 1996','免费小说',
          '\u3000', '\n', '\r','\t','。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....', '......','【','】','—','-','，','.',':',';','?','!','[',']','(',')','{','}','<','>',
          '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b', '她', '他', '你', '我', '它', '这'] #去掉其中的一些无意义的词语
        for a in ad:
            con = con.replace(a, '')
        text = re.sub(r'[^\u4e00-\u9fa5]', '', con)
    return text
def load_corpus(folder_path):
    names= os.listdir(folder_path)
    corpus=[]
    for name in names:
        novel_name = folder_path + '\\' + name
        processed_text=preprocess_text(novel_name)
        corpus.append(processed_text)
    return corpus,names
def cut_text(text, stopwords):
    words = jieba.lcut(text)
    return [word for word in words if word not in stopwords and word.strip()]        
# 生成数据
def dataset_produce():
    sent_len=100
    words=[]
    folder_path = r'D:\课程\研一下NLP\work3\novels'  # 文本文件夹路径
    corpus,names=load_corpus(folder_path)
    with open(r'D:\课程\研一下NLP\work3\stopwords.txt', 'r', encoding='utf8') as f:
        stopwords = set([word.strip('\n') for word in f.readlines()])
    processed_corpus = [cut_text(text, stopwords) for text in corpus]
    for i in range(16):
        names[i]=re.sub('.txt','',names[i])
        savepath='pickle/'+names[i]+".pickle"
        with open(savepath,'wb') as f:
            pickle.dump(processed_corpus[i],f)
        with open(savepath, 'rb') as f:
            data = pickle.load(f)
        words_len = len(data)
        sent_num = math.ceil(words_len/sent_len)
        for i in range(sent_num):
            if i == sent_num-1:
                tmp = data[i*sent_len:-1]
            else:
                tmp = data[i*sent_len:(i+1)*sent_len]
            words.append(tmp)
    with open(r'D:\课程\研一下NLP\work3\data.txt', 'w', encoding='utf-8') as f:
        data_str = '\n'.join([' '.join(row) for row in words])
        f.write(data_str)
# 计算段落语义关联
def get_paragraph_vector(paragraph, model):
    tokens = [word for word in jieba.lcut(paragraph) if word in model.wv] if isinstance(model, Word2Vec) else [word for word in jieba.lcut(paragraph) if word in model.dictionary]
    vectors = [model.wv[token] for token in tokens] if isinstance(model, Word2Vec) else [model.word_vectors[model.dictionary[token]] for token in tokens]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
def model_Word2vec():
    # sentences = list(word2vec.LineSentence('data.txt'))
    # model = Word2Vec(sentences, hs=1, min_count=5, window=5,vector_size=100, sg=1, workers=multiprocessing.cpu_count(), epochs=50)
    # model.save("word2vec——skipgram.model")

    model = Word2Vec.load(r'D:\课程\研一下NLP\work3\word2vec——skipgram.model')
    # 词语聚类
    test_name = ['李文秀', '袁承志', '胡斐', '狄云', '韦小宝', '剑客', '郭靖', '杨过'
             , '陈家洛', '萧峰', '石破天', '令狐冲', '胡斐', '张无忌', '袁冠南', '阿青','剑法','客栈'
             ,'峨嵋派','屠龙刀','蛤蟆功','葵花宝典']
    
    for name in test_name:
        print(name)
        for result in model.wv.similar_by_word(name, topn=10):
            print(result[0], '{:.6f}'.format(result[1]))
        print('-------------------------')
    # 语意距离
    print(model.wv.similarity('杨过', '小龙女'))
    print(model.wv.similarity('杨过', '东方'))
    print(model.wv.similarity('华山派', '弟子'))
    print(model.wv.similarity('华山派', '剑法'))
    print(model.wv.similarity('刀光', '剑影'))
    print(model.wv.similarity('人民', '国家'))
    print(model.wv.similarity('人民', '剑影'))

    names = [result[0] for name in test_name for result in model.wv.similar_by_word(name, topn=10)]
    name_vectors = np.array([model.wv[name] for name in names])
    tsne = TSNE()
    embedding = tsne.fit_transform(name_vectors)
    n = 6
    label = KMeans(n).fit(embedding).labels_
    plt.title('kmeans聚类结果')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(len(label)):
        if label[i] == 0:
            plt.plot(embedding[i][0], embedding[i][1], 'ro', )
        if label[i] == 1:
            plt.plot(embedding[i][0], embedding[i][1], 'go', )
        if label[i] == 2:
            plt.plot(embedding[i][0], embedding[i][1], 'yo', )
        if label[i] == 3:
            plt.plot(embedding[i][0], embedding[i][1], 'co', )
        if label[i] == 4:
            plt.plot(embedding[i][0], embedding[i][1], 'bo', )
        if label[i] == 5:
            plt.plot(embedding[i][0], embedding[i][1], 'mo', )
        plt.annotate(names[i], xy=(embedding[i][0], embedding[i][1]), xytext=(embedding[i][0]+0.1, embedding[i][1]+0.1))
    plt.show()
    plt.savefig('cluster-1.png')

    sample_paragraph1_str = '黄蓉不答，拉著他手走到後院，四下瞧了瞧，这才说道：“你和过儿的对答，我在窗外都听见啦。他不怀好意，你知道麽？”郭靖吃了一惊，问道：“甚麽不怀好意？”黄蓉道： “我听他言中之意，早在疑心咱俩害死了他爹爹。”郭靖道：“他或许确有疑心，但我已答允将他父亲逝世的情由详细说给他知道。”黄蓉道：“你当真要毫不隐瞒的说给他听？”郭靖道：“他父亲死得这麽惨，我心中一直自责。杨康兄弟虽然误入歧途，但咱们也没好好劝他，没想法子挽救。”黄蓉哼了一声，道：“这样的人又有甚麽可救的？我只恨杀他不早，否则你那几位师父又何致命丧桃花岛上？”郭靖想到这桩恨事，不禁长长叹了口气。'
    sample_paragraph2_str = '黄蓉道：“朱大哥叫芙儿来跟我说，这次过儿来到襄阳，神气中很透著点儿古怪，又说你和他同榻而眠。我担心有何意外，一直守在你窗下。我瞧还是别跟他睡在一房的好，须知人心难测，而他父亲……总是因为一掌拍在我肩头，这才中毒而死。”郭靖道：“那可不能说是你害死他的啊。”黄蓉道：“既然你我均有杀他之心，结果他也因我而死，那麽是否咱们亲自下手，也没多大分别。”郭靖沉思半晌，道：“你说得对。那麽我还是不跟他明言的为是。蓉儿，你累了半夜，快回房休息罢。过了今晚，明日我搬到军营中睡。”'
    paragraph_vector1 = get_paragraph_vector(sample_paragraph1_str, model)
    paragraph_vector2 = get_paragraph_vector(sample_paragraph2_str, model)
    paragraph_similarity = cosine_similarity([paragraph_vector1], [paragraph_vector2])[0][0]
    print(f"Semantic similarity between paragraphs: {paragraph_similarity}")

    sample_paragraph1_str = '中国共产党已走过百年奋斗历程。我们党立志于中华民族千秋伟业，致力于人类和平与发展崇高事业，责任无比重大，使命无上光荣。全党同志务必不忘初心、牢记使命，务必谦虚谨慎、艰苦奋斗，务必敢于斗争、善于斗争，坚定历史自信，增强历史主动，谱写新时代中国特色社会主义更加绚丽的华章。'
    sample_paragraph2_str = '十九大以来的五年，是极不寻常、极不平凡的五年。党中央统筹中华民族伟大复兴战略全局和世界百年未有之大变局，就党和国家事业发展作出重大战略部署，团结带领全党全军全国各族人民有效应对严峻复杂的国际形势和接踵而至的巨大风险挑战，以奋发有为的精神把新时代中国特色社会主义不断推向前进。'
    paragraph_vector1 = get_paragraph_vector(sample_paragraph1_str, model)
    paragraph_vector2 = get_paragraph_vector(sample_paragraph2_str, model)

    paragraph_similarity = cosine_similarity([paragraph_vector1], [paragraph_vector2])[0][0]
    print(f"Semantic similarity between paragraphs: {paragraph_similarity}")



if __name__ == "__main__":
    # dataset_produce()
    model_Word2vec()
