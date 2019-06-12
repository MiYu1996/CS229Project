import random
import numpy as np
import torch
from torch.autograd import Variable

class DataLoader(object):

    def __init__(
            self, src_word2idx, src_char2idx,
            src=None, src_char = None, tgt=None,
            cuda=True, batch_size=64, shuffle=True, test=False):

        self.cuda = cuda
        self.test = test
        self._n_batch = int(np.ceil(len(src) / batch_size))

        self._batch_size = batch_size

        self._src = src
        self._tgt = tgt
        self._src_char = src_char

        src_idx2word = {idx: word for word, idx in src_word2idx.items()}
        src_idx2char = {v: k for k, v in src_char2idx.items()}
        
        #Get the word to index and index to word conversion
        self._src_word2idx = src_word2idx
        self._src_idx2word = src_idx2word

        #Get the char to index and index to char conversion
        self._src_word2idx = src_char2idx
        self._src_idx2word = src_idx2char

        self._iter_count = 0

        self._need_shuffle = shuffle
        
        #Decide if there is a need to shuffle the data
        if self._need_shuffle:
            self.shuffle()

    @property
    def n_insts(self):
        return len(self._src)

    @property
    def src_vocab_size(self):
        return len(self._src_word2idx)

    @property
    def src_word2idx(self):
        return self._src_word2idx


    @property
    def src_idx2word(self):
        return self._src_idx2word


    def shuffle(self):
        if self._tgt is not None:
            paired_insts = list(zip(self._src, self._tgt))
            random.shuffle(paired_insts)
            self._src, self._tgt = zip(*paired_insts)
        else:
            random.shuffle(self._src)


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        
        def pad_to_longest(insts):

            max_len = max(len(inst) for inst in insts)

            inst_data = np.array([
                inst + [0] * (max_len - len(inst))
                for inst in insts])
            inst_data_tensor = torch.tensor(inst_data, dtype = torch.long)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.to('cuda:0')
            return inst_data_tensor
        
        def pad_char_to_longest(src, char_list):
            max_sent = src.shape[1]
            max_char = 40
            pad_list = np.zeros((len(char_list), max_sent, max_char))
            for i in range(len(char_list)):
                for j in range(len(char_list[i])):
                    if max_sent < len(char_list[i]):
                        break
                    for k in range(len(char_list[i][j])):
                        if max_char < len(char_list[i][j]):
                            break
                        else:
                            pad_list[i][j][k] = char_list[i][j][k]
            
            pad_array = np.array(pad_list)
            inst_data_tensor = torch.tensor(pad_array, dtype = torch.long)
            if self.cuda:
                inst_data_tensor = inst_data_tensor.to('cuda:0')

            return inst_data_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            src = self._src[start_idx:end_idx]
            src_data = pad_to_longest(src)
            
    
            src_char = self._src_char[start_idx:end_idx]
            src_char = pad_char_to_longest(src_data, src_char)

            if self._tgt is not None:
                tgt = self._tgt[start_idx:end_idx]
                if self.test:
                    tgt = torch.tensor(tgt, dtype = torch.float32, requires_grad = False)
                else:
                    tgt = torch.tensor(tgt, dtype = torch.float32, requires_grad = True)
            else:
                return src_data, src_char, None
            if self.cuda:
                tgt = tgt.to('cuda:0')
            return src_data, src_char, tgt

        else:

            if self._need_shuffle:
                self.shuffle()

            self._iter_count = 0
            raise StopIteration()