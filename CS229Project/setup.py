import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
import operator

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from collections import defaultdict
from sklearn.model_selection import train_test_split


import sys

#A function that use dataset path to load the training set
#and the test set

#input: directory path for the dataset folder
#output:
# train_df: dataframe for training 
# test_df: dataframe for test
def load_data(path):
    TRAIN_DATA_FILE = path + 'train.csv'
    TEST_DATA_FILE = path + 'test.csv'

    train_df = pd.read_csv(TRAIN_DATA_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)

    return train_df, test_df

#A function that load the cleaned words that help data pre-processing
#input: path for the cleanwords txt in the form typo,correct

#output: A dictionary with key typo and value corrected

def load_clean(path):
    cl_path = path
    clean_word_dict = {}
    with open(cl_path, 'r', encoding='utf-8') as cl:
        for line in cl:
            line = line.strip('\n')
            typo, correct = line.split(',')
            clean_word_dict[typo] = correct
    return clean_word_dict


#A function that splot the dataset to training_set and dev_set
#input: train_df: The original training set
#percent: the percentage of the original training set we want to put into dev_set
#The default dev set size is set to be 20%

#There is no output for this part, just save two new csv in the dataset
#folder 

def train_split(train_df, percent = 0.2):
    np.random.seed(1)
    msk = np.random.rand(len(train_df)) < percent
    train = train_df[~msk]
    dev = train_df[msk]
    return train, dev



#A function that execute the data cleaning duty
#Input: text: the comment texts
#clean_word_dict: dictionary that correct words
# remove_stopwords: If True, remove stop words
# stem_wrods: if True, turn words into their stem form
# count_null_words:  

def preprocess(text, clean_word_dict , remove_stopwords = False, stem_words = False):

    # Regex to remove all Non-Alpha Numeric and space
    special_character_removal=re.compile(r'[^?!.,:a-z\d ]',re.IGNORECASE)

    # regex to replace all numerics
    replace_numbers=re.compile(r'\d+',re.IGNORECASE)

    #Making all the words to be lower case
    #Also remove special characters and numeric values 
    text = text.lower()
    
    for typo, correct in clean_word_dict.items():
        text = re.sub(typo, " " + correct + " ", text)
    
    #expand abbreviation and separate punctuations 
    #to be a separate characters
    #This part is from last year's competition
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\!", " ! ", text)
    text = re.sub(r"\"", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    #remove the numerics
    text = replace_numbers.sub(' ', text)

    #remove the special characters
    text = special_character_removal.sub('',text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Optionally, remove stop words of the text
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        text = text.split()
        filtered_sentence = [w for w in text if not w in stop_words]
        text = " ".join(filtered_sentence)

    return (text)

#performing the data cleaning procesure on the dataset
#input: dataset_path plus / at the end
#clean_word_path: clean word text path

#This function will save cleaned data into two new csv files 
#Also will split the data into training and dev set

def clean(dataset_path, clean_word_path):
    #preprocess texts in the datasets
    train_df, test_df = load_data(dataset_path)
    clean_word_dict = load_clean(clean_word_path)

    print('*' * 25)
    print('Start pre-process the texts in the dataset!')
    print('*' * 25)

    list_sentences_train = train_df["comment_text"].fillna("no comment").values
    list_sentences_test = test_df["comment_text"].fillna("no comment").values
    

    comments = [preprocess(text,clean_word_dict) for text in list_sentences_train]    
    test_comments=[preprocess(text,clean_word_dict) for text in list_sentences_test]

    print('*' * 25)
    print('Finished pre-process the texts in the dataset!')
    print('*' * 25)


    # This save the clearned data to the corersponding directory
    train_df['comment_text'] = comments
    test_df['comment_text'] = test_comments

    #Here we split the trian to train and dev for check performance
    new_train_df, new_dev_df = train_split(train_df)


    #Now save those as well
    train_df.to_csv(dataset_path + 'cleaned_train.csv', index=False)
    test_df.to_csv(dataset_path + 'cleaned_test.csv', index=False)
    new_train_df.to_csv(dataset_path + 'cleaned_new_train.csv', index=False)
    new_dev_df.to_csv(dataset_path + 'cleaned_dev.csv', index=False)


#call the function to test
if __name__ == "__main__":
    clean('/Users/MiYu/Desktop/phd/ThirdQuarter/CS229/Projects/ProjectCode/dataset/', '/Users/MiYu/Desktop/phd/ThirdQuarter/CS229/Projects/ProjectCode/features/cleanwords.txt')
    
    #Use to create a tiny set
    #train_df = pd.read_csv('/Users/MiYu/Desktop/phd/ThirdQuarter/CS229/Projects/ProjectCode/dataset/cleaned_train.csv')
    #_, tiny = train_split(train_df, percent = 0.001)
    #tiny_train, tiny_dev = train_split(tiny, percent = 0.2)
    #tiny_train.to_csv('/Users/MiYu/Desktop/phd/ThirdQuarter/CS229/Projects/ProjectCode/dataset/tiny_train.csv')
    #tiny_dev.to_csv('/Users/MiYu/Desktop/phd/ThirdQuarter/CS229/Projects/ProjectCode/dataset/tiny_dev.csv')




