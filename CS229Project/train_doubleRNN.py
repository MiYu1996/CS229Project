import argparse
import time
import os
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from nn_models import GMP, GRUCnn, bLSTM2DCNN, doubleRNN
from data_loader import DataLoader
from utils import *
from tensorboardX import SummaryWriter


import numpy as np
from sklearn.model_selection import KFold
#This is for the tensorboard visualization


def train(name, model, training_data, validation_data, crit, optimizer, scheduler, opt, index_val):
    
    valid_aucs = [0.]
    #This is for the record of generalized AUC
    general_aucs = [0.]
    writer = SummaryWriter()
    ite = 0
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')
        start = time.time()
        train_loss, train_auc, ite = train_epoch(model, training_data, crit, optimizer, ite, writer)
        print('  - (Training)   loss: {loss: 8.5f}, auc: {auc:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  loss=train_loss, auc=100*train_auc,
                  elapse=(time.time()-start)/60))
        
        
        start = time.time()
        valid_loss, valid_auc, valid_proba = eval_epoch(model, validation_data, crit)
        
        #This part is to used to calculate the generalized AUC
        #read in the validation set
        df1 = pd.read_csv('dataset/cleaned_new_train.csv')
        df1 = df1.iloc[index_val, :]
        df1['prediction'] = valid_proba
        general_auc = final_metric(df1, 'All', 0.25, 0.25, 0.25, 0.25)
        writer.add_scalar('Generalized AUC', general_auc, epoch_i)
        ####


        print('  - (Validation) loss: {loss: 8.5f}, auc: {auc:3.3f}, Generalized AUC: {G_auc:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    loss=valid_loss, auc=100*valid_auc, G_auc = 100*general_auc,
                    elapse=(time.time()-start)/60))
        
        #This was used for last year's competition 
        #best_loss = max(valid_aucs)
        best_loss = max(general_aucs)
        valid_aucs += [valid_auc]
        general_aucs += [general_auc]
        scheduler.step(valid_loss)

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i,
            'auc': valid_auc,
            'G_auc': general_auc}

        model_name = name + '.chkpt'
        if general_auc >= best_loss:
            print('new best loss:', general_auc)
            best_proba = valid_proba
            best_model = model
            if opt.save_model:
                torch.save(checkpoint, 'models/'+model_name)
                print('    - [Info] The checkpoint file has been updated.')

        if opt.log:
            directory = 'predictions/' + opt.name
            log_train_file = directory + '/train.log'
            log_valid_file = directory + '/valid.log'

            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{fold},{epoch},{loss: 8.5f},{auc:3.3f}\n'.format(
                    fold=name, epoch=epoch_i, loss=train_loss, auc=100*train_auc))
                log_vf.write('{fold},{epoch},{loss: 8.5f},{auc:3.3f},{G_auc:3.3f}\n'.format(
                    fold=name, epoch=epoch_i, loss=valid_loss, auc=100*valid_auc, G_auc = 100*general_auc))
    writer.close()

    return best_model, best_proba


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True, help='Path to preprocessed data')
    parser.add_argument('-name', required=True, help='Name of experiment')

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=512)

    parser.add_argument('-dropout', type=float, default=0.3)

    parser.add_argument('-log', action='store_true')
    parser.add_argument('-save_model', action='store_true')
    parser.add_argument('-word2vec', action='store_true')
    parser.add_argument('-fasttext', action='store_true')
    parser.add_argument('-concat_embed', action='store_true')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-char', action='store_true')
    parser.add_argument('-one_fold', action='store_true', help='Train single fold')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # ========= Loading Dataset =========#
    data = torch.load(opt.data)
    if opt.word2vec:
        embeds = data['word2vec']
    elif opt.fasttext:
        embeds = data['fasttext']
    elif opt.concat_embed:
        embed1 = data['word2vec']
        embed2 = data['fasttext']
        embeds = np.concatenate((embed1, embed2), axis = 1)
        print("Successfully concat the two embeddings, now embedding shape: ", embeds.shape)
    
    #load the character embedding dictionary 
    n_src_char = data['dict_char']

    opt.max_token_seq_len = data['settings'].max_token_seq_len
    opt.src_vocab_size = len(data['dict'])

    print(opt)
    
    directory = 'predictions/'+opt.name
    if not os.path.exists(directory):
        os.makedirs(directory)

    if opt.log:
        log_train_file = directory + '/train.log'
        log_valid_file = directory + '/valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('fold,epoch,loss,auc\n')
            log_vf.write('fold,epoch,loss,auc\n')

    crit = nn.BCEWithLogitsLoss()

    if opt.cuda:
        crit = crit.cuda()

    # =======================================#
    n_train = len(data['train']['src'])
    idx = list(range(n_train))
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    test_data = DataLoader(
        data['dict'],
        data['dict_char'],
        src=np.array(data['test']['src']),
        src_char = np.array(data['test']['src_char']),
        batch_size=opt.batch_size,
        shuffle=False,
        test=True,
        cuda=opt.cuda)

    out = {}
    i = 0
    # ========= Preparing CrossVal =========#
    for idx_train, idx_val in kf.split(idx):

        #model = GMP(opt.src_vocab_size, embeds=embeds, dropout=opt.dropout)
        #model = GRUCnn(opt.src_vocab_size, embeds=embeds, dropout=opt.dropout)
        # model = bLSTM2DCNN(opt.src_vocab_size, embeds=embeds)
        if opt.char:
            mode = 'char'
            print('Using character embedding and word embedding!')
        else:
            mode = 'word'
            print('Using word embedding!')

        model = GRUCnn(opt.src_vocab_size, n_src_char, embeds=embeds, mode = mode)
        
        if opt.cuda:
            model = model.cuda()
        
        optimizer = optim.Adam(model.get_trainable_parameters(), lr=0.001,
                                betas=(0.9, 0.98), eps=1e-09, weight_decay=5e-5, )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=2, verbose=1)

        # ========= DataLoaders =========#
        training_data = DataLoader(
            data['dict'],
            data['dict_char'],
            src=np.array(data['train']['src'])[idx_train],
            src_char = np.array(data['train']['src_char'])[idx_train],
            tgt=np.array(data['train']['tgt'])[idx_train],
            batch_size=opt.batch_size,
            cuda=opt.cuda)

        validation_data = DataLoader(
            data['dict'],
            data['dict_char'],
            src=np.array(data['train']['src'])[idx_val],
            src_char = np.array(data['train']['src_char'])[idx_val],
            tgt=np.array(data['train']['tgt'])[idx_val],
            batch_size=opt.batch_size,
            shuffle=False,
            test=True,
            cuda=opt.cuda)

        fold_name = str(opt.name) + '_fold_'+ str(i)

        model_ft, val_proba = train(fold_name, model, training_data,
                                    validation_data, crit, optimizer, scheduler, opt, idx_val)

        out[i] = {
            'idx': idx_val,
            'proba': val_proba

        }

        df_out = create_submit_df(model_ft, dataloader=test_data)
        df_out.to_csv(directory+'/'+fold_name+'_test.csv', index=False)

        if opt.one_fold:
            break
        i += 1

    torch.save(out, directory+'/cv.prob')


if __name__ == '__main__':
    main()