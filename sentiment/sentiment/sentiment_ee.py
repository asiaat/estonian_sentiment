'''
Created on 18 Jul 2017

@author: malle
'''
import sklearn
import pandas
import _pickle as cPickle
import csv
from textblob.classifiers import NaiveBayesClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from textblob import TextBlob
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold,  train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.externals import joblib
#from sklearn.learning_curve import learning_curve

train = [
   ('I love this sandwich.', 'pos'),
   ('this is an amazing place!', 'pos'),
   ('I feel very good about these beers.', 'pos'),
   ('this is my best work.', 'pos'),
   ("what an awesome view", 'pos'),
   ('I do not like this restaurant', 'neg'),
   ('I am tired of this stuff.', 'neg'),
   ("I can't deal with this", 'neg'),
   ('he is my sworn enemy!', 'neg'),
   ('my boss is horrible.', 'neg'),
   ('This movie is  shit!.', 'neg')
 ]

test = [
     ('the beer was good.', 'pos'),
     ('I do not enjoy my job', 'neg'),
     ("I ain't feeling dandy today.", 'neg'),
     ("I feel amazing!", 'pos'),
     ('Gary is a friend of mine.', 'pos'),
     ("I can't believe I'm doing this.", 'neg')
 ]

def classify(inp_text):
    cl  = NaiveBayesClassifier(train)
    res = cl.classify(inp_text)
    print (cl.accuracy(test))
    print (cl.show_informative_features(5)) 
    return res


def read_data(inp_file):
    messages = [line.rstrip() for line in open(inp_file)]
    print (len(messages))
    
    messages = pandas.read_csv(inp_file, sep='\t', 
                               #quoting=csv.QUOTE_NONE,
                           names=["label", "message"])
    return messages
    
def split_into_lemmas(inp_message):
    #message = str(inp_message, 'utf8').lower()
    message = inp_message.lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]



    
    
    

def make_detector(inp_file):
    #res = classify("This movie is a piece of  shit!")
    #print (res)
    
    messages = read_data('./data/laused.tsv')
    print(messages)
        
    bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
    print (len(bow_transformer.vocabulary_))
    
    message4 = messages['message'][3]
    print (message4)
    
    bow4 = bow_transformer.transform([message4])
    print (bow4)
    print (bow4.shape)
    
    messages_bow = bow_transformer.transform(messages['message'])
    print( 'sparse matrix shape:', messages_bow.shape)
    print( 'number of non-zeros:', messages_bow.nnz)
    print( 'sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])))
    
    
    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    tfidf4 = tfidf_transformer.transform(bow4)
    print (tfidf4)
   
    
    messages_tfidf = tfidf_transformer.transform(messages_bow)
    print (messages_tfidf.shape)
    
    spam_detector = MultinomialNB().fit(messages_tfidf, messages['label'])
    print ('predicted:', spam_detector.predict(tfidf4)[0])
    print ('expected:', messages.label[3])
    
    
    all_predictions = spam_detector.predict(messages_tfidf)
    print ('accuracy', accuracy_score(messages['label'], all_predictions))
    print ('confusion matrix\n', confusion_matrix(messages['label'], all_predictions))

    print (classification_report(messages['label'], all_predictions))
    
    
    msg_train, msg_test, label_train, label_test = \
    train_test_split(messages['message'], messages['label'], test_size=0.2)

    print (len(msg_train), len(msg_test), len(msg_train) + len(msg_test))
    
    
    pipeline_svm = Pipeline([
        ('bow', CountVectorizer(analyzer=split_into_lemmas)),
        ('tfidf', TfidfTransformer()),
        ('classifier', SVC()),  # <== change here
    ])
    
    # pipeline parameters to automatically explore and tune
    param_svm = [
      {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
      {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
    ]
    
    grid_svm = GridSearchCV(
        pipeline_svm,  # pipeline from above
        param_grid=param_svm,  # parameters to tune via cross validation
        refit=True,  # fit using all data, on the best detected classifier
        n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
        scoring='accuracy',  # what score are we optimizing?
        cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
    )
    
    
    svm_detector = grid_svm.fit(msg_train, label_train) # find the best combination from param_svm
    
    print (svm_detector.predict(["Persse kõik!"])[0])
    print (svm_detector.predict(["See on hea restoran!"])[0])
    print (svm_detector.predict(["esmaselt tuli uuesti tulla"])[0])
    
    # store the spam detector to disk after training
    #with open('sms_spam_detector.pkl', 'wb') as fout:
    #    cPickle.dump(svm_detector, fout)
    
    joblib.dump(svm_detector, inp_file) 
    

def use_detector(inp_file):
    
    
    messages = read_data('./data/laused.tsv')
    print(messages)
        
    bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
    print (len(bow_transformer.vocabulary_))
    
    message4 = "Laulupidu on üks tore üritus!"
    m5 = "mul on  halb olla"
    
    # ...and load it back, whenever needed, possibly on a different machine
    #svm_detector_reloaded = cPickle.load(open('sms_spam_detector.pkl'))
    svm_detector_reloaded = joblib.load(inp_file)
    
    #print ('before:', svm_detector.predict([message4])[0])
    print ('after:', svm_detector_reloaded.predict([message4])[0])
    print ('after:', svm_detector_reloaded.predict([m5])[0])
    
  
if __name__ == '__main__':
    
    #messages = read_data('./data/toksents.tsv')
    #print(messages)
    
    file_model = 'filename5.pkl'
    #make_detector(file_model)
    use_detector(file_model)
    