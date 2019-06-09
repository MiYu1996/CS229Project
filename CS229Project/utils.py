import pandas as pd
import numpy as np
import nltk
from tqdm import tqdm
from sklearn import metrics 
from sklearn.metrics import roc_auc_score
import pickle
import os
import math
import shutil
import torch.nn.functional as F

import torch
from torch.autograd import Variable
from torch.nn import init


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from scipy.sparse import hstack




#This is the function that calculates the overall AUC 

#input: df: A dataframe with column 'target' with 0 and 1 lebel
# and 0 as non-toxic also with column 'prediction' with 0 and 1 lebel

#output: the overall AUC metric

def auc(df):
    y = df['target'] 
    pred = df['prediction'] 
    fpr, tpr, thresholds = metrics.roc_curve(y,pred)
    return metrics.auc(fpr, tpr)



#p-th power function
#Take the average of the power function and then take the corresponding roots
#input: data: the data we are taking average at
#p: the p-th power, default -5.0 was given by the competition 
def Mp(data, p = -5.0):
    return np.average(data ** p) ** (1/p)

#This is the function that will returns theS SUB, BPSN, and BNSP for the group of identities given
#  SUB : Claculates an AUC using only examples from the sub-group.
#  BPSN : Background Positive Subgroup Negative. Claculates an AUC using a subset of toxic comments 
#   outside the sub-group (BP) and non-toxic comments in the sub-group (SN).
#  BNSP : Background Negative Subgroup Positive. Claculates an AUC using a subset of non-toxic comments 
# outside the sub-group (BN) and toxic comments in the sub-group (SP). 

#input: groups: the list of identities that we will calculate the metrics on
#df: A dataframe with column 'target' with probability
# and 0 as non-toxic also with column 'prediction' with probability

#output: return a data frame containing all the sub-metrics

def SUB_BPSN_BNSP(groups, df):
    if groups == 'All': 
        #These are the whole identities 
        categoriese = ['buddhist','hindu','atheist','intellectual_or_learning_disability','other_gender','physical_disability','bisexual','heterosexual',
        'other_disability','other_sexual_orientation','transgender','latino','psychiatric_or_mental_illness',
        'jewish','asian','homosexual_gay_or_lesbian','other_religion','other_race_or_ethnicity','black','muslim','white','christian','female','male']
    else:
        categoriese = groups
    categoriese_df = pd.DataFrame(columns = ['SUB','BPSN','BNSP'], index = categoriese)

    for category in categoriese:
        #change it to 0 or 1 rather than probabilities
        #if the identity is mentioned or not
        #if the category is NA, treated it as 0
        df[category] = df[category] >= 0.5
        #calculate the subgroup AUC
        #it is possible that there is no data, then we will just assign each value to be 0
        if df[df[category]].shape[0] == 0:
            #remove the entire row
             categoriese_df = categoriese_df.drop(category, axis = 0)
        else:
            categoriese_df.loc[category,'SUB'] = auc(df[df[category]])
            bpsn = ((~df[category] & df['target'])    #background positive
                | (df[category] & ~df['target'])) #subgroup negative
            categoriese_df.loc[category,'BPSN'] = auc(df[bpsn])
            bnsp = ((~df[category] & ~df['target'])   #background negative
                | (df[category] & df['target']))  #subgrooup positive
            categoriese_df.loc[category,'BNSP'] = auc(df[bnsp])
    #drop rows that contain NANs due to insufficient data
    categoriese_df = categoriese_df.dropna(axis = 0)

    #Apply the power function defined before
    categoriese_df.loc['Mp',:] = categoriese_df.apply(Mp, axis= 0)

    return categoriese_df


#input: df: A dataframe with column 'target' with probability
#groups: the list of identities that we will calculate the metrics on

#w0, w1, w2, w3: weights for each sub-metrics. In this project, this should be set to be
# 0.25 


#output: This output our final metric 
def final_metric(df,groups,w0,w1,w2,w3):
    #First of all, swtich the probability to labels
    df['target'] = df['target'] >= 0.5
    df['prediction']  = df['prediction'] >= 0.5  

    #Next caluclate the overall arc and the Mp of each sub-metrics
    overall = auc(df)
    categoriese_df = SUB_BPSN_BNSP(groups, df)
    final_metric = w0 * overall + w1 * categoriese_df.loc['Mp','SUB'] + w2 * categoriese_df.loc['Mp','BPSN'] + w3 * categoriese_df.loc['Mp','BNSP']

    return final_metric




#This function uses TfidfVEctorizer to extract the features from
#given corpus, we can then use these features to do
#non-nn methods

#input: train_df: the training dataframe
#test_df: the test dataframe
#train_path: the path for original cleaned training data
#test_path: the path for original cleaned test data 

#output: train_features: the features for the training dataset


#therefore all_text is the text of the train and the test
def feature_extractor(train_df, test_df, train_path, test_path):
    #initialize the word n-gram extractor

    #first of all, concatenate all the comments
    orig_train = pd.read_csv(train_path)
    orig_test = pd.read_csv(test_path)

    print('Finish loading data!')


    #list_sentence_train = train_df['comment_text'].values.astype('U')
    list_sentence_train = train_df['comment_text'].apply(lambda x: np.str(x))
    list_sentence_test = test_df['comment_text'].apply(lambda x: np.str(x))
    list_sentence_train_orig = orig_train['comment_text']
    list_sentence_test_orig = orig_test['comment_text']
    all_text = pd.concat([list_sentence_train_orig,list_sentence_test_orig]).apply(lambda x: np.str(x))

    #creating the iterable containing the comments
    train_text = []
    test_text = []
    for word in list_sentence_train:
        train_text.append(word)
    
    for word in list_sentence_test:
        test_text.append(word)



    word_vectorizer = TfidfVectorizer(
        #replace tf with 1 + log(tf)
        #couple researches has shown that this
        #works better
        sublinear_tf = True,
        #remove accents 'unicode' works on
        #any character
        strip_accents = 'unicode',
        analyzer = 'word',
        token_pattern = r'\w{1,}',
        #we use 1-gram and 2-gram features
        ngram_range = (1,2),
        #number of features, for here let us set it to be 20000
        max_features = 20000
        )
    #fit the word_vectorizer
    #Now transoform the data into the document-term matrix
    #This takes iterable, so we make a list containing all the
    #comments
    word_vectorizer.fit(all_text)
    print('Finish fitting word vectorizer!')

    train_word_features = word_vectorizer.transform(train_text)
    test_word_features = word_vectorizer.transform(test_text)
    
    #save the result
    pickle.dump(word_vectorizer, open("/Users/MiYu/Desktop/phd/ThirdQuarter/CS229/Projects/ProjectCode/features/word_tfidf.pickle", "wb"))
    pickle.dump(train_word_features, open("/Users/MiYu/Desktop/phd/ThirdQuarter/CS229/Projects/ProjectCode/features/train_word_features.pickle", "wb"))
    pickle.dump(test_word_features, open("/Users/MiYu/Desktop/phd/ThirdQuarter/CS229/Projects/ProjectCode/features/test_word_features.pickle", "wb"))

    print('Finish Extracting both word features!')

    #Now let us get the character features
    #We think that for toxic comment detection
    #character feature can be extremely
    #important

    char_vectorizer = TfidfVectorizer(
        #replace tf with 1 + log(tf)
        #couple researches has shown that this
        #works better
        sublinear_tf = True,
        #remove accents 'unicode' works on
        #any character
        strip_accents = 'unicode',
        analyzer = 'char',
        #we use 1-gram to 4-gram features
        ngram_range = (1,6),
        #number of features, for here let us set it to be 30000
        max_features = 30000
        )
    #fit the charcter features to the text
    #and transform the 
    char_vectorizer.fit(all_text)
    print('Finish fitting char vectorizer!')
    train_char_features = char_vectorizer.transform(train_text)
    test_char_features = char_vectorizer.transform(test_text)

    #save the result
    #load result use pickle.load()
    pickle.dump(char_vectorizer, open("/Users/MiYu/Desktop/phd/ThirdQuarter/CS229/Projects/ProjectCode/features/char_tfidf.pickle", "wb"))
    #Too larget so cannot dump these 
    #pickle.dump(train_char_features, open("/Users/MiYu/Desktop/phd/ThirdQuarter/CS229/Projects/ProjectCode/features/train_char_features.pickle", "wb"))
    #pickle.dump(test_char_features, open("/Users/MiYu/Desktop/phd/ThirdQuarter/CS229/Projects/ProjectCode/features/test_char_features.pickle", "wb"))

    print('Finish Extracting both character features!')
    
    #stack the training features with the
    #test features to get the features 
    train_features = hstack([train_char_features, train_word_features])
    test_features = hstack([test_char_features, test_word_features])

    return train_features, test_features



#This function is used to evaluation 
#when given a prediction_csv and the original dataset that
#the prediction was given

#input: data_set_path: notice that this dataset must contain the true label
#in the feature name 'target'
# prediction_path: the path that contains probability_csv

#output: The evaluation score for the model
def evaluation(data_set_path, prediction_path):
    df1 = pd.read_csv(data_set_path)
    df2 = pd.read_csv(prediction_path)
    #if the category is NA, treated it as 0
    merged = df1.merge(df2, on = "id", how = 'left').fillna(0)
    #Use all default from the competition
    return final_metric(merged, 'All', 0.25, 0.25, 0.25, 0.25)


###The rest are helper function for training neural network



def train_epoch(model, training_data, crit, optimizer):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_total = 0
    probs = []
    tgts = []

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        src, tgt = batch
        tgt = torch.unsqueeze(tgt, 1)

        # forward
        optimizer.zero_grad()
        pred = model(src)
        proba = F.sigmoid(pred).data.cpu().numpy()

        # backward
        loss = crit(pred, tgt)
        loss.backward()
        optimizer.step()

        n_total += 1
        total_loss += loss.data
        tgts.append(tgt.data.cpu().numpy())
        probs.append(proba)

    tgts = np.vstack(tgts)
    probs = np.vstack(probs)

    auc = np.mean(roc_auc_score(tgts, probs))

    return total_loss/n_total, auc


def eval_epoch(model, validation_data, crit):

    model.eval()

    total_loss = 0
    n_total = 0
    probs = []
    tgts = []

    for batch in tqdm(
            validation_data, mininterval=2,
            desc='  - (Validation) ', leave=False):

        src, tgt = batch
        tgt = torch.unsqueeze(tgt, 1)

        # forward
        pred = model(src)
        loss = crit(pred, tgt)
        proba = F.sigmoid(pred).data.cpu().numpy()

        n_total += 1
        total_loss += loss.data
        tgts.append(tgt.data.cpu().numpy())
        probs.append(proba)

    tgts = np.vstack(tgts)
    probs = np.vstack(probs)
    auc = np.mean(roc_auc_score(tgts, probs))

    return total_loss/n_total, auc, probs


def create_submit_df(model, dataloader):

    df = pd.read_csv('data/test.csv', usecols=['id'])
    classes = ['target']
    df = df.reindex(columns=['id'] + classes)

    model.eval()
    probs = []

    for batch in tqdm(
            dataloader, mininterval=2,
            desc='  - (Creating submission file) ', leave=False):

        src, *_ = batch

        pred = model(src)
        proba = F.sigmoid(pred).data.cpu().numpy()
        probs.append(proba)

    probs = np.vstack(probs)
    df[classes] = probs
    print('    - [Info] The submission file has been created.')
    return df














