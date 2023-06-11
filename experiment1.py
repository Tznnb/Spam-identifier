# 导入所需的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# 读取数据集
spam_data = pd.read_csv("spamassassin.csv")
enron_data = pd.read_csv("enron.csv")


# 对数据集进行预处理，去除空白符、标点符号、数字、停用词等无关信息，将所有字母转换为小写，进行词干提取和词形还原等
def preprocess(text):
    # TODO: implement the preprocessing function
    return text


spam_data["Email"] = spam_data["Email"].apply(preprocess)
enron_data["Email"] = enron_data["Email"].apply(preprocess)


# 对数据集进行特征提取，使用词袋模型或TF-IDF作为权重计算方法，将每封电子邮件表示为一个由不同单词组成的向量，每个单词对应一个权重，反映该单词在电子邮件中出现的频率或重要性
def feature_extraction(texts, method, max_features):
    # 创建向量化器对象，根据method参数选择词袋模型或TF-IDF作为权重计算方法，并设置最大特征数为max_features
    if method == "bag-of-words":
        vectorizer = TfidfVectorizer(use_idf=False, max_features=max_features)
    elif method == "tf-idf":
        vectorizer = TfidfVectorizer(max_features=max_features)
    else:
        raise ValueError("Invalid method")
    # 对文本进行向量化
    features = vectorizer.fit_transform(texts)
    # 返回特征矩阵和特征名称
    return features, vectorizer.get_feature_names()


# 构建分类器，使用多项式朴素贝叶斯分类器作为分类器，根据贝叶斯定理和条件独立性假设，计算每封电子邮件属于垃圾邮件或正常邮件的概率，然后根据最大后验概率原则进行分类
def classifier_build(alpha):
    # 创建多项式朴素贝叶斯分类器对象，并设置拉普拉斯平滑参数α
    classifier = MultinomialNB(alpha=alpha)
    # 返回分类器对象
    return classifier


# 数据集划分，将数据集按照80%和20%的比例划分为训练集和测试集
def dataset_split(features, labels, ratio):
    # 获取数据集的总数
    n = features.shape[0]
    # 获取训练集的数量
    m = int(n * ratio)
    # 随机打乱数据集的顺序
    indices = np.random.permutation(n)
    # 划分训练集和测试集
    train_features = features[indices[:m]]
    train_labels = labels[indices[:m]]
    test_features = features[indices[m:]]
    test_labels = labels[indices[m:]]
    # 返回训练集和测试集
    return train_features, train_labels, test_features, test_labels


# 不同参数对性能的影响：分析不同max_features值对性能的影响。图1展示了在不同max_features值下，使用SpamAssassin Corpus作为数据集时，垃圾邮件标识器在测试集上的准确率变化情况。图2展示了在不同max_features值下，使用Enron-Spam Corpus作为数据集时，垃圾邮件标识器在测试集上的准确率变化情况。
def max_features_experiment(max_features_values, method, data):
    # 创建空列表，用于存储不同max_features值下的准确率
    accuracy_list = []
    # 遍历不同的max_features值
    for max_features in max_features_values:
        # 对数据集进行特征提取，根据method参数选择词袋模型或TF-IDF作为权重计算方法，并设置最大特征数为max_features
        features, feature_names = feature_extraction(data["Email"], method, max_features)
        # 对数据集进行划分，按照80%和20%的比例划分为训练集和测试集
        train_features, train_labels, test_features, test_labels = dataset_split(features, data["Label"], 0.8)
        # 构建分类器，并设置拉普拉斯平滑参数α为1
        classifier = classifier_build(1)
        # 使用训练集对分类器进行训练
        classifier.fit(train_features, train_labels)
        # 使用测试集对分类器进行评估，获取准确率
        accuracy = accuracy_score(test_labels, classifier.predict(test_features))
        # 将准确率添加到列表中
        accuracy_list.append(accuracy)
        # 打印当前max_features值和准确率
        print("max_features:", max_features)
        print("accuracy:", accuracy)
        print()

    # 绘制图表，展示不同max_features值下的准确率变化情况
    plt.plot(max_features_values, accuracy_list, label=method)
    plt.xlabel("max_features")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


# 定义不同max_features值的列表，从1000到10000，间隔为1000
max_features_values = np.arange(1000, 11000, 1000)

# 运行实验函数，分析不同max_features值对性能的影响，并绘制图1和图2
max_features_experiment(max_features_values, "bag-of-words", spam_data)
max_features_experiment(max_features_values, "tf-idf", spam_data)
max_features_experiment(max_features_values, "bag-of-words", enron_data)
max_features_experiment(max_features_values, "tf-idf", enron_data)


# 不同方法对性能的影响：分析不同的特征提取方法对性能的影响。表1展示了使用词袋模型和TF-IDF时，在SpamAssassin Corpus上的垃圾邮件标识器在测试集上的准确率、召回率、精确率和F1值。表2展示了使用词袋模型和TF-IDF时，在Enron-Spam Corpus上的垃圾邮件标识器在测试集上的准确率、召回率、精确率和F1值。
def method_experiment(methods, data):
    # 创建空列表，用于存储不同方法下的评估指标
    accuracy_list = []
    recall_list = []
    precision_list = []
    f1_list = []
    # 遍历不同的方法
    for method in methods:
        # 对数据集进行特征提取，根据method参数选择词袋模型或TF-IDF作为权重计算方法，并设置最大特征数为5000
        features, feature_names = feature_extraction(data["Email"], method, 5000)
        # 对数据集进行划分，按照80%和20%的比例划分为训练集和测试集
        train_features, train_labels, test_features, test_labels = dataset_split(features, data["Label"], 0.8)
        # 构建分类器，并设置拉普拉斯平滑参数α为1
        classifier = classifier_build(1)
        # 使用训练集对分类器进行训练
        classifier.fit(train_features, train_labels)
        # 使用测试集对分类器进行评估，获取评估指标
        accuracy = accuracy_score(test_labels, classifier.predict(test_features))
        recall = recall_score(test_labels, classifier.predict(test_features), pos_label="spam")
        precision = precision_score(test_labels, classifier.predict(test_features), pos_label="spam")
    f1 = f1_score(test_labels, classifier.predict(test_features), pos_label="spam")
    # 将评估指标添加到列表中
    accuracy_list.append(accuracy)
    recall_list.append(recall)
    precision_list.append(precision)
    f1_list.append(f1)
    # 打印当前方法和评估指标
    print("method:", method)
    print("accuracy:", accuracy)
    print("recall:", recall)
    print("precision:", precision)
    print("f1:", f1)
    print()

    # 绘制图表，展示不同方法下的评估指标


x = np.arange(len(methods))
width = 0.2
plt.bar(x - width / 2, accuracy_list, width, label="accuracy")
plt.bar(x + width / 2, recall_list, width, label="recall")
plt.bar(x + 3 * width / 2, precision_list, width, label="precision")
plt.bar(x + 5 * width / 2, f1_list, width, label="f1")
plt.xticks(x + width / 2, methods)
plt.ylabel("score")
plt.legend()
plt.show()

# 定义不同方法的列表，包括词袋模型和TF-IDF
methods = ["bag-of-words", "tf-idf"]

# 运行实验函数，分析不同方法对性能的影响，并绘制表1和表2
method_experiment(methods, spam_data)
method_experiment(methods, enron_data)


# 不同方法的对比：本文还与其他方法进行了对比，以验证垃圾邮件标识器的优势。表3展示了不同方法在SpamAssassin Corpus上的准确率、召回率、精确率和F1值。表4展示了不同方法在Enron-Spam Corpus上的准确率、召回率、精确率和F1值。
def method_comparison(methods, data):
    # 创建空列表，用于存储不同方法下的评估指标
    accuracy_list = []
    recall_list = []
    precision_list = []
    f1_list = []
    # 遍历不同的方法
    for method in methods:
        # 根据方法的名称，选择不同的特征提取方法和分类器
        if method == "Naive Bayes with TF-IDF":
            features, feature_names = feature_extraction(data["Email"], "tf-idf", 5000)
            classifier = classifier_build(1)
        elif method == "Support Vector Machine with TF-IDF":
            features, feature_names = feature_extraction(data["Email"], "tf-idf", 5000)
            classifier = SVC()
        elif method == "Logistic Regression with TF-IDF":
            features, feature_names = feature_extraction(data["Email"], "tf-idf", 5000)
            classifier = LogisticRegression()
        else:
            raise ValueError("Invalid method")
        # 对数据集进行划分，按照80%和20%的比例划分为训练集和测试集
        train_features, train_labels, test_features, test_labels = dataset_split(features, data["Label"], 0.8)
        # 使用训练集对分类器进行训练
        classifier.fit(train_features, train_labels)
        # 使用测试集对分类器进行评估，获取评估指标
        accuracy = accuracy_score(test_labels, classifier.predict(test_features))
        recall = recall_score(test_labels, classifier.predict(test_features), pos_label="spam")
    precision = precision_score(test_labels, classifier.predict(test_features), pos_label="spam")
    f1 = f1_score(test_labels, classifier.predict(test_features), pos_label="spam")
    # 将评估指标添加到列表中
    accuracy_list.append(accuracy)
    recall_list.append(recall)
    precision_list.append(precision)
    f1_list.append(f1)
    # 打印当前方法和评估指标
    print("method:", method)
    print("accuracy:", accuracy)
    print("recall:", recall)
    print("precision:", precision)
    print("f1:", f1)
    print()

    # 绘制图表，展示不同方法的对比


x = np.arange(len(methods))
width = 0.2
plt.bar(x - width / 2, accuracy_list, width, label="accuracy")
plt.bar(x + width / 2, recall_list, width, label="recall")
plt.bar(x + 3 * width / 2, precision_list, width, label="precision")
plt.bar(x + 5 * width / 2, f1_list, width, label="f1")
plt.xticks(x + width / 2, methods)
plt.ylabel("score")
plt.legend()
plt.show()

# 定义不同方法的列表，包括朴素贝叶斯、支持向量机、逻辑回归等
methods = ["Naive Bayes with TF-IDF", "Support Vector Machine with TF-IDF", "Logistic Regression with TF-IDF"]

# 运行实验函数，进行不同方法的对比，并绘制表3和表4
method_comparison(methods, spam_data)
method_comparison(methods, enron_data)

