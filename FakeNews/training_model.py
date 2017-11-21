import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
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

    path_real = 'res/mail_online/'

    paper = newspaper.build('http://www.dailymail.co.uk/news/index.html', memoize_articles=False)
    paper_name = 'MAIL_ONLINE'
    limit = NUM_OF_ARTICLES if NUM_OF_ARTICLES < len(paper.articles) else len(paper.articles)
    for i in range(limit):
        article = paper.articles[i]
        save_article(article, path_real, paper_name, i)


def parse_fake_news():

    path_fake = 'res/new_data/'
    file = 'res/fake_links'
    with open(file, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            try:
                article = newspaper.Article(lines[i])
                article.download()
                article.parse()
            except:
                print("Couldnt download article")
                continue
            if len(article.title) > 0 and len(article.text) > 0:
                save_article(article, path_fake, "SOME FAKE ARTICLE", i)


def size_of_data():
    tot_num_of_articles = 0
    buzzfeed = len(os.listdir('res/buzzfeed'))
    tot_num_of_articles += len(os.listdir('res/buzzfeed'))
    # print(buzzfeed)
    tot_num_of_articles += len(os.listdir('res/times'))
    tot_num_of_articles += len(os.listdir('res/washington'))
    tot_num_of_articles += len(os.listdir('res/fox_news'))
    tot_num_of_articles += len(os.listdir('res/huffington'))
    tot_num_of_articles += len(os.listdir('res/nbc'))
    tot_num_of_articles += len(os.listdir('res/google'))
    tot_num_of_articles += len(os.listdir('res/yahoo'))
    tot_num_of_articles += len(os.listdir('res/abc'))
    tot_num_of_articles += len(os.listdir('res/bbc'))
    print(tot_num_of_articles)

    print(len(os.listdir('res/data/fake_news')))


def size_of_labeled():
    print("fake_news: " + str(len(os.listdir('res/data/fake_news'))))
    print("real_news: " + str(len(os.listdir('res/data/real_news'))))


def predict(present_graph):

    # Get the data
    articles = load_files('res/data')

    # Create the word to vector transformers
    count_vect = CountVectorizer(ngram_range=(1, 3))
    X_train_counts = count_vect.fit_transform(articles.data)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


    if present_graph:
        plot_decision_graph(X_train_tfidf, articles)

    # Train the classifier
    # clf = MultinomialNB().fit(X_train_tfidf, articles.target)
    # clf = SGDClassifier().fit(X_train_tfidf, articles.target)
    # clf = LogisticRegression().fit(X_train_tfidf, articles.target)
    # clf = svm.SVC().fit(X_train_tfidf, articles.target)
    clf = svm.LinearSVC().fit(X_train_tfidf, articles.target)

    # Now predict
    docs_new_names = []
    docs_new_content = []
    path = 'res/new_data/'

    num_test_files = len(os.listdir(path))

    for file_name in os.listdir(path):
        with open(path + file_name) as f:
            if "fake" in file_name.lower():
                label = 'fake_news'
            else:
                label = 'real_news'
            docs_new_names.append((file_name, label))
            docs_new_content.append(f.read())


    X_new_counts = count_vect.transform(docs_new_content)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)
    prob_of_pred = clf.decision_function(X_new_tfidf)

    # Print results
    correct = 0
    i = 0
    predicted_fake = 0
    predicted_fake_correctly = 0
    predicted_real = 0
    predicted_real_correctly = 0

    for doc, category in zip(docs_new_names, predicted):
        label = articles.target_names[category]
        print('%r => %s\t\t%f' % (doc[0], label, prob_of_pred[i]))

        if label == 'fake_news':
            predicted_fake += 1
        else:
            predicted_real += 1

        if doc[1] == label:
            correct += 1
            if label == 'fake_news':
                predicted_fake_correctly += 1
            else:
                predicted_real_correctly += 1
        i += 1

    print("Predicted fake: " + str(predicted_fake))
    print("Predicted fake correctly: " + str(predicted_fake_correctly))
    print("Predicted real: " + str(predicted_real))
    print("Predicted real correctly: " + str(predicted_real_correctly))
    print("Score: " + str(correct / num_test_files))
    print("On: " + str(num_test_files) + " articles, 25 fake and 25 real.")


def get_precision():
    # Get the data
    articles = load_files('res/data')

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


def plot_decision_graph(X_, data):

    # import some data to play with
    X = X_[:, :2] # we only take the first two features. We could
    # avoid this ugly slicing by using a two-dim dataset
    y = data.target

    h = .02  # step size in the mesh

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
    lin_svc = svm.LinearSVC(C=C).fit(X, y)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel']

    for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        x_first = np.array(X[:, 0])
        x_last = np.array(X[:, 1])
        plt.scatter(x_first, x_last, c=y, cmap=plt.cm.coolwarm)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.show()


if __name__ == '__main__':
    # parse_news()

    # parse_fake_news()

    start = time.process_time()
    predict(False)
    print(time.process_time() - start)
