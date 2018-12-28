import random
import csv
import gensim
import ast
import pickle
from numpy import array, mean
from gensim.test.utils import datapath
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

from sklearn.externals import joblib


def createTrainingDataset():
    GISWDataLength = 3000
    GIDataLength = 3000
    count = 0
    corpus = []
    print("Extracting training data: ")
    with open('cleanedGeneralIssuesCommonTS1toTS2.csv', mode='r') as commonReader:
        commonReader.readline()
        csvGISWReader = csv.reader(commonReader, delimiter=',')
        for row in csvGISWReader:
            string = row[5]
            string = [str(x) for x in string.strip().split()]
            if len(string) <= 1:
                continue
            corpus.append([1, row[1], string])
            count += 1
            if count > GISWDataLength:
                break
    for i in corpus[0:5]:
        print(i)

    count = 0
    with open('cleanedGeneralIssuesTS1.csv', mode='r') as generalReader:
        generalReader.readline()
        csvGIReader = csv.reader(generalReader, delimiter=',')
        for row in csvGIReader:
            string = row[5]
            string = [str(x) for x in string.strip().split()]
            if len(string) <= 1 or row[1] == "[deleted]":
                continue
            corpus.append([0, row[1], string])
            count += 1
            if count > GIDataLength:
                break
    for i in corpus[-5:]:
        print(i)

    random.shuffle(corpus)
    print("SHUFFLING->>>>>>>>>>>>>")
    for i in corpus[0:5]:
        print(i)
    for i in corpus[-5:]:
        print(i)

    with open('training.csv', mode='w') as writer:
        csvWriter = csv.writer(writer, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerow(['Class', 'Author', 'Text'])
        for row in corpus:
            csvWriter.writerow(row)
    print("Done Extracting")
    pass


def createTestingDataset():
    GISWDataLength = 672
    GIDataLength = 1800
    count = 0
    corpus = []
    print("Extracting testing data: ")
    with open('cleanedGeneralIssuesCommonTS1toTS2.csv', mode='r') as commonReader:
        commonReader.readline()
        csvGISWReader = csv.reader(commonReader, delimiter=',')
        for row in csvGISWReader:
            string = row[5]
            string = [str(x) for x in string.strip().split()]
            if len(string) <= 1:
                continue

            count += 1
            if (count < 1120):
                continue
            corpus.append([1, row[1], string])
            if count > GISWDataLength + 1120:
                break
    for i in corpus[0:5]:
        print(i)

    count = 0
    with open('cleanedGeneralIssuesTS1.csv', mode='r') as generalReader:
        generalReader.readline()
        csvGIReader = csv.reader(generalReader, delimiter=',')
        for row in csvGIReader:
            string = row[5]
            string = [str(x) for x in string.strip().split()]
            if len(string) <= 1 or row[1] == "[deleted]":
                continue
            count += 1
            if (count < 3000):
                continue
            corpus.append([0, row[1], string])
            if count > GIDataLength + 3000:
                break
    for i in corpus[-5:]:
        print(i)

    random.shuffle(corpus)
    print("SHUFFLING->>>>>>>>>>>>>")
    for i in corpus[0:5]:
        print(i)
    for i in corpus[-5:]:
        print(i)

    with open('testing.csv', mode='w') as writer:
        csvWriter = csv.writer(writer, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerow(['Class', 'Author', 'Text'])
        for row in corpus:
            csvWriter.writerow(row)
    print("Done Extracting")
    pass


def createValidationDataset():
    GISWDataLength = 448
    GIDataLength = 1200
    count = 0
    corpus = []
    print("Extracting validation data: ")
    with open('cleanedGeneralIssuesCommonTS1toTS2.csv', mode='r') as commonReader:
        commonReader.readline()
        csvGISWReader = csv.reader(commonReader, delimiter=',')
        for row in csvGISWReader:
            string = row[5]
            string = [str(x) for x in string.strip().split()]
            if len(string) <= 1:
                continue

            count += 1
            if (count < 1120 + 672):
                continue
            corpus.append([1, row[1], string])
            if count > GISWDataLength + 1120 + 672:
                break
    for i in corpus[0:5]:
        print(i)

    count = 0
    with open('cleanedGeneralIssuesTS1.csv', mode='r') as generalReader:
        generalReader.readline()
        csvGIReader = csv.reader(generalReader, delimiter=',')
        for row in csvGIReader:
            string = row[5]
            string = [str(x) for x in string.strip().split()]
            if len(string) <= 1 or row[1] == "[deleted]":
                continue
            count += 1
            if (count < 3000 + 1800):
                continue
            corpus.append([0, row[1], string])
            if count > GIDataLength + 3000 + 1800:
                break
    for i in corpus[-5:]:
        print(i)

    random.shuffle(corpus)
    print("SHUFFLING->>>>>>>>>>>>>")
    for i in corpus[0:5]:
        print(i)
    for i in corpus[-5:]:
        print(i)

    with open('validation.csv', mode='w') as writer:
        csvWriter = csv.writer(writer, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerow(['Class', 'Author', 'Text'])
        for row in corpus:
            csvWriter.writerow(row)
    print("Done Extracting")
    pass


def trainingModel():
    corpus = []
    docNumber = 0
    with open('training.csv', mode='r') as reader:
        reader.readline()
        csvReader = csv.reader(reader)
        for row in csvReader:
            corpus.append([str(x) for x in row[2].strip().split()])
            docNumber += 1

    print("Total Count: ", docNumber)
    dictionary = gensim.corpora.Dictionary(corpus)
    count = 0
    for k, v in dictionary.iteritems():
        print(k, v)
        count += 1
        if count > 10:
            break
    print(len(dictionary))
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=3000)
    count = 0
    for k, v in dictionary.iteritems():
        print(k, v)
        count += 1
        if count > 10:
            break

    bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]
    print(bow_corpus[6])

    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=2, id2word=dictionary, passes=2, workers=4)
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

    # save the model to disk
    modelFile = datapath('/Users/kushalgevaria/PycharmProjects/Capstone/ldaTrainedModel')
    lda_model.save(modelFile)
    pass


def bestTopicModel():
    searchParams = {'n_components': [n for n in range(5, 30)], 'learning_decay': [.5, .7, .9]}
    lda = LatentDirichletAllocation()
    model = GridSearchCV(lda, searchParams)

    corpus = []
    docNumber = 0
    with open('training.csv', mode='r') as reader:
        reader.readline()
        csvReader = csv.reader(reader)
        for row in csvReader:
            if (int(row[0]) == 1):
                corpus.append(" ".join(ast.literal_eval(row[2])))
                docNumber += 1
    print("Total Doc: ", docNumber)
    for i in corpus[0:5]:
        print(i)

    vectorizer = CountVectorizer(min_df=10)
    data_vectorized = vectorizer.fit_transform(corpus)
    print('Started Training')
    model.fit(data_vectorized)
    print('Found the best model')
    # Best Model
    best_lda_model = model.best_estimator_

    # Model Parameters
    print("Best Model's Params: ", model.best_params_)

    # Log Likelihood Score
    print("Best Log Likelihood Score: ", model.best_score_)

    # Perplexity
    print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))
    filename = 'finalized_LDAModel.sav'
    pickle.dump(best_lda_model, open(filename, 'wb'))

    pass



def getTrainFeatures(filename):
    corpus = []
    docNumber = 0
    suicidalCount = 0
    nonsuicidalCount = 0
    with open(filename, mode='r') as reader:
        reader.readline()
        csvReader = csv.reader(reader)
        for row in csvReader:
            corpus.append(array([row[0], " ".join(ast.literal_eval(row[2]))]))
            if (row[0] == '1'):
                suicidalCount += 1
            else:
                nonsuicidalCount += 1
            docNumber += 1
    print("Total Doc: ", docNumber)
    print("Suicidal Count: ", suicidalCount)
    print("Non-Suicidal Count: ", nonsuicidalCount)
    corpus = array(corpus)
    # print(corpus[:,1])

    vectorizer = CountVectorizer(min_df=10)
    data_vectorized = vectorizer.fit_transform(corpus[:, 1])
    print(data_vectorized.shape)
    print(vectorizer.vocabulary_.get('depress'))

    tf_transformer = TfidfTransformer(use_idf=False).fit(data_vectorized)
    X_train_tf = tf_transformer.transform(data_vectorized)
    print(X_train_tf.shape)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(data_vectorized)
    print(X_train_tfidf.shape)

    return X_train_tfidf, corpus[:, 0], vectorizer, tfidf_transformer

def getTestFeatures(filename):
    corpus = []
    docNumber = 0
    suicidalCount = 0
    nonsuicidalCount = 0
    with open(filename, mode='r') as reader:
        reader.readline()
        csvReader = csv.reader(reader)
        for row in csvReader:
            corpus.append(array([row[0], " ".join(ast.literal_eval(row[2]))]))
            if(row[0] == '1'):
                suicidalCount += 1
            else:
                nonsuicidalCount += 1
            docNumber += 1
    print("Total Doc: ", docNumber)
    print("Suicidal Count: ", suicidalCount)
    print("Non-Suicidal Count: ", nonsuicidalCount)
    corpus = array(corpus)
    return corpus
    # print(corpus[:,1])

    # vectorizer = CountVectorizer(min_df=10)
    # data_vectorized = vectorizer.transform(corpus[:, 1])
    # print(data_vectorized.shape)
    # print(vectorizer.vocabulary_.get('depress'))
    #
    # tf_transformer = TfidfTransformer(use_idf=False).fit(data_vectorized)
    # X_train_tf = tf_transformer.transform(data_vectorized)
    # print(X_train_tf.shape)
    #
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(data_vectorized)
    # print(X_train_tfidf.shape)

    # return X_train_tfidf, corpus[:, 0]


def SVM():
    X_train_tfidf, target, countVectorizer, tfidf_transformer = getTrainFeatures('training.csv')
    print('Started Training...')
    # clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=500, tol=None)
    clf = MultinomialNB()
    clf.fit(X_train_tfidf, target)
    print('Done Training...')
    print('Started Testing...')

    testingCorpus = getTestFeatures('testing.csv')
    X_test_counts = countVectorizer.transform(testingCorpus[:, 1])
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    predictions = clf.predict(X_test_tfidf)


    # print(len(test_target))
    # predictions = clf.predict(X_test_tfidf)
    print(predictions)
    # print(test_target)
    print(mean(predictions == testingCorpus[:, 0]))
    print(confusion_matrix(testingCorpus[:, 0], predictions))
    print(confusion_matrix(testingCorpus[:, 0], predictions).ravel())

    validatingCorpus = getTestFeatures('validation.csv')
    X_test_counts = countVectorizer.transform(validatingCorpus[:, 1])
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    predictions = clf.predict(X_test_tfidf)

    # print(len(test_target))
    # predictions = clf.predict(X_test_tfidf)
    print(predictions)
    # print(test_target)
    print(mean(predictions == validatingCorpus[:, 0]))
    print(confusion_matrix(validatingCorpus[:, 0], predictions))
    print(confusion_matrix(validatingCorpus[:, 0], predictions).ravel())


def main():
    # createTrainingDataset()
    # createTestingDataset()
    # createValidationDataset()
    # trainingModel()
    # bestTopicModel()
    # SVM()
    pass


if __name__ == '__main__':
    main()
