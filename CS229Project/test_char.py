import argparse
import torch
import torch.nn as nn
from utils import *
from data_loader import DataLoader
from nn_models import *

def load_model(model_name, modelA):
    checkpoint = torch.load('models/'+model_name)
    modelA.load_state_dict(checkpoint['model'])
    
    return modelA


def main(): 
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True, help='Path to preprocessed data')
    parser.add_argument('-FILE', required=True, help='checkpointname')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-batch_size', type=int, default=512)
    parser.add_argument('-word2vec', action='store_true')
    parser.add_argument('-fasttext', action='store_true')
    parser.add_argument('-concat_embed', action='store_true')

    parser.add_argument('-dropout', type=float, default=0.3)

    opt = parser.parse_args()


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
    opt.max_token_seq_len = data['settings'].max_token_seq_len
    opt.src_vocab_size = len(data['dict'])
    n_src_char = data['dict_char']

    opt.cuda = not opt.no_cuda


    model = GRUCnn(opt.src_vocab_size, n_src_char, embeds=embeds, dropout=opt.dropout, mode = 'char')
    if opt.cuda:
            model = model.cuda()
    model = load_model(opt.FILE, model)



    #load the test_data
    print('Loading test Data')
    test_data = DataLoader(
        data['dict'],
        data['dict_char'],
        src=np.array(data['test']['src']),
        src_char = np.array(data['test']['src_char']),
        batch_size=opt.batch_size,
        shuffle=False,
        test=True,
        cuda=opt.cuda)
    
    print('Finish Loading Test Data')
    
    print('Start prediction!')
    df_out = create_submit_df(model, dataloader=test_data)
    df_out.to_csv('predictions/'+ opt.FILE +'_test.csv', index=False)
    print('Prediction .csv File Generated')
    
    print('Calculating the Metric')
    df1 = pd.read_csv('dataset/cleaned_new_test.csv')
    df1['prediction'] = df_out['target']
    general_auc = final_metric(df1, 'All', 0.25, 0.25, 0.25, 0.25)
    print('The final metric is ' + str(general_auc))


if __name__ == '__main__':
    main()