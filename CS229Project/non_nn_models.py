import numpy as np
import pandas as pd
import pickle

import numpy as np
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_val_predict
from scipy.sparse import hstack
import scipy.sparse

import setup
import utils



#This is the first non-neural
#baseline model we will try

#this object will train logtistic regresison as well as
#naive bayes

class baseline:
    def __init__(self, train_df, test_df): 
        #store the two dataframe
        print('*' * 25)
        print('start Initializing!')
        print('*' * 25)
        self.train_df = train_df
        self.test_df = test_df
        self.train_features = None
        self.test_features = None
        print('*' * 25)
        print('finish Initializing!')
        print('*' * 25)
    
    #This function need to be called to get the features
    #mode can be pre-trined or train
    #path 1: if mode  ='train', this should be the training file path for the model to extract features
    #if mode = 'pre-trained', this should be the test file path for the model to extract word feature
    #path 2:  if mode  ='train', this should be the test file path for the model to extract features
    #if mode = 'pre-trained', this should be the test file path for the model to extract char feature
    def _extract(self, path1, path2, mode):
        if mode == 'train':
            print('*' * 25)
            print('start Extracting!')
            print('*' * 25)
            self.train_features, self.test_features = utils.feature_extractor(self.train_df, self.test_df, path1, path2)

            print('*' * 25)
            print('Finish Extracting!')
            print('*' * 25)
        
        elif mode == 'pre-trained':
            print('*' * 25)
            print('start Extracting the pretrained-vectorizer!')
            print('*' * 25)
            word_vectorizer = pickle.load(open(path1,'rb'))
            char_vectorizer = pickle.load(open(path2, 'rb'))

            print('*' * 25)
            print('Finish Extracting the pretrained-vectorizer!')
            print('*' * 25)

            
            print('*' * 25)
            print('start extracting the features on word and char!')
            print('*' * 25)
            list_sentence_train = self.train_df['comment_text'].apply(lambda x: np.str(x))
            list_sentence_test = self.test_df['comment_text'].apply(lambda x: np.str(x))


            train_text = []
            test_text = []
            for word in list_sentence_train:
                train_text.append(word)
    
            for word in list_sentence_test:
                test_text.append(word)


            train_word_features = word_vectorizer.transform(train_text)
            test_word_features = word_vectorizer.transform(test_text)


            train_char_features = char_vectorizer.transform(train_text)
            test_char_features = char_vectorizer.transform(test_text)

            self.train_features = hstack([train_char_features, train_word_features])
            self.test_features = hstack([test_char_features, test_word_features])

            print('*' * 25)
            print('finish extracting the features on word and char!')
            print('*' * 25)



    
    

       
    def _train_model(self, mode):
        assert mode in ['Logistic', 'NB', 'SVM']
        #train the logistic regression model
        #And will save the prediction for data to the save_path specified
        if mode == 'Logistic':
            print('*' * 25)
            print('start training!')
            print('*' * 25)
        
            prediction = {'id': self.test_df['id']}
            train_target = self.train_df['target'] >= 0.5


            print('*' * 25)
            print('start training Logistic Regression Model!')
            print('*' * 25)
            #This part is for the logistic regression
            classifier = LogisticRegression(solver = 'sag')
            classifier.fit(self.train_features,train_target)

            #save the model to the disk
            filename = 'finalized_logistic_model.sav'
            pickle.dump(classifier, open(filename, 'wb'))
            print('*' * 25)
            print('Logistic Regression Model Saved Successfully!')
            print('*' * 25)

        
            print('*' * 25)
            print('Start Prediction for Logistic Regression!')
            print('*' * 25)
            prediction['prediction'] = classifier.predict_proba(self.test_features)[:, 1]
            prediction = pd.DataFrame.from_dict(prediction)
            prediction.to_csv('Logistic_prediction.csv', index=False)
            print('*' * 25)
            print('Prediction for Logistic Regression Saved Successfully!')
            print('*' * 25)

            del prediction
            del classifier

        elif mode == 'NB':
            #Reinitialize the prediction 
            prediction_naive = {'id': self.test_df['id']}
            train_target = self.train_df['target'] >= 0.5

            print('*' * 25)
            print('start training Naive Bayes Model!')
            print('*' * 25)
            #This part is for the logistic regression
            classifier = MultinomialNB()
            classifier.fit(self.train_features,train_target)

            #save the model to the disk
            filename = 'finalized_naive_bayes_model.sav'
            pickle.dump(classifier, open(filename, 'wb'))
            print('*' * 25)
            print('Naive Bayes Model Saved Successfully!')
            print('*' * 25)

        
            print('*' * 25)
            print('Start Prediction!')
            print('*' * 25)
            prediction_naive['prediction'] = classifier.predict_proba(self.test_features)[:, 1]
            prediction_naive = pd.DataFrame.from_dict(prediction_naive)
            prediction_naive.to_csv('naive_prediction.csv', index=False)
            print('*' * 25)
            print('Prediction for Naive Bayes Saved Successfully!')
            print('*' * 25)
        
        else:
            #Reinitialize the prediction
            prediction_svm = {'id': self.test_df['id']}
            train_target = self.train_df['target'] >= 0.5

            print('*' * 25)
            print('start training SVM model!')
            print('*' * 25)
            #This part is for the logistic regression
            classifier = svm.SVC(gamma = 'scale', probability= True)
            classifier.fit(self.train_features,train_target)

            #save the model to the disk
            filename = 'finalized_svm.sav'
            pickle.dump(classifier, open(filename, 'wb'))
            print('*' * 25)
            print('SVM Model Saved Successfully!')
            print('*' * 25)

        
            print('*' * 25)
            print('Start Prediction!')
            print('*' * 25)
            prediction_svm['prediction'] = classifier.predict_proba(self.test_features)[:, 1]
            prediction_svm = pd.DataFrame.from_dict(prediction_svm)
            prediction_svm.to_csv('svm_prediction.csv', index=False)
            print('*' * 25)
            print('Prediction for SVM Saved Successfully!')
            print('*' * 25)



        
        

        print('*' * 25)
        print('Finished!')
        print('*' * 25)
    

    #This creates folders for cross validation
    #This function currently does not work
  

    def train_folders(self, fold_count):
        fold_size = self.train_df.shape[0] // fold_count
        
        prediction = {'id': self.test_df['id']}
        for fold_id in range(0, fold_count):
            fold_start = fold_size * fold_id
            fold_end = fold_start + fold_size

            if fold_id == fold_size - 1:
                fold_end = self.train_df.shape[0]

            train_x = np.concatenate([self.train_features[:fold_start], self.train_features[fold_end:]])
            train_y = np.concatenate([self.train_df[:fold_start]['target'] >= 0.5, self.train_df[fold_end:]['target'] >= 0.5])
            
            val_x = self.train_features[fold_start:fold_end]


            classifier = LogisticRegression(solver = 'sag')
            classifier.fit(train_x,train_y)
            prediction[str(fold_id)] = classifier.predict_proba(val_x)[:, 1]

            print("In fold #", fold_id)
        
        #Write the list to a csv file
        prediction = pd.DataFrame.from_dict(prediction)
        prediction.to_csv('Logistic_prediction_cv.csv', index=False)
        



if __name__ == "__main__":
    ### test run for the Logistic regression
    #Try on training and dev set

    train_path = '/Users/MiYu/Desktop/phd/ThirdQuarter/CS229/Projects/ProjectCode/dataset/cleaned_new_train.csv'
    test_path = '/Users/MiYu/Desktop/phd/ThirdQuarter/CS229/Projects/ProjectCode/dataset/cleaned_dev.csv'
    
    #Since we already extracted and saved the feature, we just need to load the extracted features
    #model._extract(word_vectorizer_path, char_vectorizer_path, mode = 'pre-trained')

    ### test run for tiny set
    #test_path = '/Users/MiYu/Desktop/phd/ThirdQuarter/CS229/Projects/ProjectCode/dataset/tiny_dev.csv'
    #train_path = '/Users/MiYu/Desktop/phd/ThirdQuarter/CS229/Projects/ProjectCode/dataset/tiny_train.csv'
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    word_vectorizer_path = '/Users/MiYu/Desktop/phd/ThirdQuarter/CS229/Projects/ProjectCode/features/word_tfidf.pickle'
    char_vectorizer_path = '/Users/MiYu/Desktop/phd/ThirdQuarter/CS229/Projects/ProjectCode/features/char_tfidf.pickle' 
    
    model = baseline(train_df, test_df)
    model._extract(word_vectorizer_path, char_vectorizer_path, mode = 'pre-trained')
    model._train_model(mode = 'SVM')


    #prediction_path_Logistic = '/Users/MiYu/Desktop/phd/ThirdQuarter/CS229/Projects/ProjectCode/Logistic_prediction.csv'
    #prediction_path_naive = '/Users/MiYu/Desktop/phd/ThirdQuarter/CS229/Projects/ProjectCode/naive_prediction.csv'
    prediction_path_svm = '/Users/MiYu/Desktop/phd/ThirdQuarter/CS229/Projects/ProjectCode/svm_prediction.csv'

    #print('The final metric for Logistic Regression is ' + str(utils.evaluation(test_path,prediction_path_Logistic)))

    #print('The final metric for Naive Bayes is ' + str(utils.evaluation(test_path,prediction_path_naive)))

    print('The final metric for Naive Bayes is ' + str(utils.evaluation(test_path,prediction_path_svm)))


    

    #side notes: the corss-validation does not work
    #currently it gives error that the coo-matrix is not subscriptable 
    #test cv
    #model.train_folders(3)
  





        
   
   

    

        

    


