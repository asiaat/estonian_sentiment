'''
Created on 19 Jul 2017

@author: Kalle Olumets
'''

import argparse
from sklearn.externals import joblib
from textblob import TextBlob

def split_into_lemmas(inp_message):
    #message = str(inp_message, 'utf8').lower()
    message = inp_message.lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Twitter scraping ')
       
    parser.add_argument('--sentence',    action="store",         dest="sentence")
       
    arguments =  parser.parse_args()
    #print (arguments.sentence)
    
    file_model = 'trainedmodel.pkl'
    svm = joblib.load(file_model)
    
    res = svm.predict([arguments.sentence])[0]
      
    
    print (arguments.sentence,res)