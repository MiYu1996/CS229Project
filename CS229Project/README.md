## Description for Unbiased Comment Toxicity Classifiation Model with NLP ##

data_loader.py: data loader that samples minibatch for training for the neural network methods

dataset:
       new_train_terget.txt: this is the sampled training target 
       sample_submission.csv: this is a sample of what a submission should look like for Kaggle Challenge
       tiny_dev.csv: a tiny dev set to test implementation
       tiny_train.csv: a tiny training set to test implementation

features:
       cleanwords.txt: this file will be used to convert characters in the original dataset to clean the data format

layers.py: implementation of the layers for neural methods

nn_models.py: implementation of all the neural network method (GMP method, GRUCnn, Self-Attention, GRUCnn with Attention, BiLSTMCNN, DoubleCNN, etc.)

non_nn_models.py: implementation of all the non-neural methods: logtistic baseline, Naive Bayes, and SVM and training for these methods

preprocess_withChar.py: preprocessing file with character embedding for neural method

preprocess.py: preprocess file of word embedding

setup.py: file that cleans the original data to a cleaned format

test_char.py: test the character embedding implementation

train_doubleRNN.py: file that trains the doubleRNN model

train.py: file that trains all the other neural methods

utils.py: includes utility functions such as evaluation metrics


