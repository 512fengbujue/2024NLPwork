import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import os
import jieba
def preprocess_text(con):
    tokens = list(jieba.cut(con)) # 按词拆分
    # tokens = list(con) # 按字拆分
    stopwords = set()  # 停用词集合，需要根据实际情况进行填充
    with open("stopwords.txt", "r", encoding="utf-8") as file:
        for line in file:
            word = line.strip()  # 去除行末尾的换行符和空格
            stopwords.add(word)  # 将停用词添加到集合中
    tokens = [token for token in tokens if token not in stopwords]
    tokens = [token for token in tokens if token != '\n']  # 去除换行符
    tokens = [token for token in tokens if token != '\u3000']  # 去除全角空格
    tokens = [token for token in tokens if token.strip() != '']  # 去除空格
    tokens = [token for token in tokens if token != '\r']  # 去除回车符
    tokens = [token for token in tokens if token != '[']  # 去除左方括号
    tokens = [token for token in tokens if token != ']']  # 去除右方括号
    return tokens
def extract_paragraphs(folder_path, K):
    corpus = []
    labels = []
    names = os.listdir(folder_path)
    for name in names:
        novel_name = folder_path + '\\' + name
        with open(novel_name, 'r', encoding='ANSI') as f:
            con = f.read()
            con = con.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')
            con = con.replace('新语丝电子文库(www.xys.org)','')
            con = con.replace('新语丝电子文库','')
            con = con.replace('Last Updated: Saturday, November 16, 1996','')
            con = preprocess_text(con)
            pos = int(len(con) // 63)  #16篇文章，分词后，每篇均匀选取63个k词段落
            for i in range(63):
                corpus.append(list(con[i * pos:i * pos + K]))
                labels.append(name[:-4])
                if len(corpus) == 1000:
                    break
        f.close()
    labels = labels[:1000]
    return corpus, labels
def lda_classify(K,T,num_cross,analyzer,X_train,X_test,y_train,y_test,classifiers,results):
    # 将文本转换为主题分布的流水线
    lda_pipeline = Pipeline([
        ('vectorizer', CountVectorizer(max_features=K, analyzer=analyzer)),
        ('lda', LatentDirichletAllocation(n_components=T, random_state=42, n_jobs=-1))
    ])
    # 将文本转换为主题分布
    X_train_lda = lda_pipeline.fit_transform([' '.join(x) for x in X_train])
    X_test_lda = lda_pipeline.transform([' '.join(x) for x in X_test])
    # 使用不同的分类器进行训练和评估
    for classifier_name, classifier in classifiers.items():
        classifier.fit(X_train_lda, y_train)
        accuracy = np.mean(cross_val_score(classifier, X_train_lda, y_train, cv=num_cross))
        test_accuracy = accuracy_score(y_test, classifier.predict(X_test_lda))
        # 保存结果
        results.append({
            'K': K,
            'T': T,
            'Classifier': classifier_name,
            'Analyzer': analyzer,
            'Training Accuracy': accuracy,
            'Test Accuracy': test_accuracy
        })
        print(K,T,classifier_name,analyzer,accuracy,test_accuracy)
    return results
# 主函数
def main():
    folder_path = r'D:\\课程\\研一下NLP\\work2\\novels'  # 文本文件夹路径
    # 定义不同的 K
    K_values = [20, 100, 500, 1000, 3000]
    # 定义不同的主题数量 T
    T_values = [5, 10, 20, 30, 50, 100, 200, 300, 500, 1000]
    # 定义交叉验证的次数
    num_cross = 10
    # 定义分类器
    classifiers = {
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "Multinomial Naive Bayes": MultinomialNB(),
        "KNN": KNeighborsClassifier()
    }
    # 定义结果存储列表
    results = []
    for K in K_values:
        for T in T_values:
            [corpus, labels] = extract_paragraphs(folder_path, K)
            X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.1, random_state=42)
            results=lda_classify(K,T,num_cross=num_cross,analyzer='word',X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,classifiers=classifiers,results=results)
            # results=lda_classify(K,T,num_cross=num_cross,analyzer='char',X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,classifiers=classifiers,results=results)
    results_df = pd.DataFrame(results)
    results_df.to_excel("result_ci.xlsx", index=False)
if __name__ == "__main__":
    main()