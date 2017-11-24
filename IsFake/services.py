import os
import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV  # This is the svm
import sklearn.svm as svm
from sklearn import metrics
import newspaper, time

NUM_OF_ARTICLES = 100
MIN_ARTICLE_LEN = 1000


def save_article(article, path, paper_name, file_name):
    try:
        article.download()
        article.parse()
    except:
        return
    title = article.title
    content = article.text
    if title is None or content is None:
        return
    if len(content) < MIN_ARTICLE_LEN:
        return
    file_name = path + paper_name + '_' + str(file_name)
    f = open(file_name, 'w')
    f.write(title)
    f.write('\n')
    f.write(content)


def parse_news():

    path_real = './res/mail_online/'

    paper = newspaper.build('http://www.dailymail.co.uk/news/index.html', memoize_articles=False)
    paper_name = 'MAIL_ONLINE'
    limit = NUM_OF_ARTICLES if NUM_OF_ARTICLES < len(paper.articles) else len(paper.articles)
    for i in range(limit):
        article = paper.articles[i]
        save_article(article, path_real, paper_name, i)


def parse_fake_news():

    path_fake = './res/data/fake_news/'
    file = './res/fake_links'
    with open(file, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            try:
                paper = newspaper.build(lines[i], memoize_articles=False)
            except:
                continue
            if len(paper.articles) > 0:
                article = paper.articles[0]
                save_article(article, path_fake, "SOME_FAKE_SITE", i)


def get_article(article):
    try:
        article.download()
        article.parse()
    except:
        return None, None
    title = article.title
    content = article.text
    if title is None or content is None:
        return None, None
    if len(content) < MIN_ARTICLE_LEN:
        return None, None
    return title, content


def parse_article(url):
    article = newspaper.Article(url)
    title, content = get_article(article)
    if title is not None and content is not None:
        return title, content
    else:
        return "error while parsing"


def size_of_data():
    tot_num_of_articles = 0
    buzzfeed = len(os.listdir('./res/buzzfeed'))
    tot_num_of_articles += len(os.listdir('./res/buzzfeed'))
    # print(buzzfeed)
    tot_num_of_articles += len(os.listdir('./res/times'))
    tot_num_of_articles += len(os.listdir('./res/washington'))
    tot_num_of_articles += len(os.listdir('./res/fox_news'))
    tot_num_of_articles += len(os.listdir('./res/huffington'))
    tot_num_of_articles += len(os.listdir('./res/nbc'))
    tot_num_of_articles += len(os.listdir('./res/google'))
    tot_num_of_articles += len(os.listdir('./res/yahoo'))
    tot_num_of_articles += len(os.listdir('./res/abc'))
    tot_num_of_articles += len(os.listdir('./res/bbc'))
    print(tot_num_of_articles)

    print(len(os.listdir('./res/data/fake_news')))


def size_of_labeled():
    print("fake_news: " + str(len(os.listdir('./res/data/fake_news'))))
    print("real_news: " + str(len(os.listdir('./res/data/real_news'))))


def predict(count_vect, tfidf_transformer, clf, articles, data):
    X_new_counts = count_vect.transform([data])
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)
    prob_of_pred = clf.decision_function(X_new_tfidf)

    label = articles.target_names[predicted[0]]
    pred_score = prob_of_pred[0]
    return label, pred_score


def get_data():

    # Get the data
    dir_path = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(dir_path)
    data_path = main_dir + '/res/data'
    print('\n' + data_path + '\n')
    articles = load_files(data_path)
    return articles


def get_trained_machine(present_graph=False):

    articles = get_data()

    # Create the word to vector transformers
    count_vect = CountVectorizer(ngram_range=(1, 3))
    X_train_counts = count_vect.fit_transform(articles.data)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Train the classifier
    # clf = MultinomialNB().fit(X_train_tfidf, articles.target)
    # clf = SGDClassifier().fit(X_train_tfidf, articles.target)
    # clf = LogisticRegression().fit(X_train_tfidf, articles.target)
    # clf = svm.SVC(C=1.0).fit(X_train_tfidf, articles.target)
    clf = svm.LinearSVC(C=1.0).fit(X_train_tfidf, articles.target)

    return count_vect, tfidf_transformer, clf, articles


def get_precision():

    articles = get_data()

    # Create the word to vector transformers
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(articles.data)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Train the classifier
    clf = SGDClassifier().fit(X_train_tfidf, articles.target)

    docs_test = articles.data

    X_test_counts = count_vect.transform(docs_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    predicted = clf.predict(X_test_tfidf)

    print(np.mean(predicted == articles.target))
    print(metrics.classification_report(articles.target, predicted, target_names=articles.target_names))
