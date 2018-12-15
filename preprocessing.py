from __future__ import unicode_literals, print_function, division
from io import open
import io
import unicodedata
import string
import re
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pdb
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.switch_backend('agg')

batch_size = 32
SOS_token = 0
EOS_token = 1
PAD_token = 2
UNK_token = 3
MAX_LENGTH = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Create Lang class that represents the input and output languages. 
# It builds vocabulary, index2word, word2index dictionaries for each language.
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<SOS>": 0, "<EOS>": 1, "<pad>": 2, "<unk>": 3}
        self.word2count = {"<SOS>": 0, "<EOS>": 0, "<pad>": 0, "<unk>": 0}
        self.index2word = {0: "<SOS>", 1: "<EOS>", 2: "<pad>", 3: "<unk>"}
        self.n_words = 4  # Count SOS, EOS, pad and unk

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1
            
    def buildVocab(self, count, train = False, in_out = False): 
        if train & in_out:
            del_list = []
            for k,v in self.word2count.items():
                if v <= count:
                    del_list.append(k)
            for k in del_list:
                self.word2count.pop(k)

        for k,v in self.word2count.items():
            self.word2index[k] = self.n_words
            self.index2word[self.n_words] = k
            self.n_words += 1

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# Delete extra space between characters in Chinese language
def normalizeZh(s):
    s = s.strip()
    s = re.sub("\s+", " ", s)
    return s

# Cut sentence length to MAX_LENGTH
def filterPair(p):
    filtered = []
    for i in p:
        filtered.append(' '.join(i.split()[:MAX_LENGTH-1]))
    return filtered

def filterPairs(pairs):
    return [filterPair(pair) for pair in pairs]

# Read in each language
# Output: input Lang class, output Lang class, all pairs of input and output sentences
def readLangs(dataset, lang1, lang2, infrequent_count):
    chinese = os.getcwd()+'/iwslt-zh-en/{}.tok.{}'.format(dataset, lang1)
    english = os.getcwd()+'/iwslt-zh-en/{}.tok.{}'.format(dataset, lang2)

    chinese_lines = open(chinese, encoding='utf-8').read().strip().split('\n')
    english_lines = open(english, encoding='utf-8').read().strip().split('\n')
    length = len(chinese_lines)

    pairs = [[normalizeZh(chinese_lines[i]), normalizeString(english_lines[i])] for i in range(length)]
    pairs = filterPairs(pairs)
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    
    if dataset == 'train':
        input_lang.buildVocab(count=infrequent_count, train=True, in_out=True)
        output_lang.buildVocab(count=infrequent_count, train=True, in_out=True)
    else:
        input_lang.buildVocab(count=infrequent_count)
        output_lang.buildVocab(count=infrequent_count)
    return input_lang, output_lang, pairs

# Load all pre-trained embeddings for input and output languages
def load_embedding(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for index, line in enumerate(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = [float(i) for i in tokens[1:]]
    return data

# Create weight matrix of input and output languages according to their vocabulary
def create_weight(index2word, embedding):
    emb_dim = 300
    words_found = 0
    wnf = []
    matrix_len = len(index2word.keys())
    weight_matrix= np.zeros((matrix_len, emb_dim))
    for k,v in index2word.items():
        if k == PAD_token:
            pass
        else:
            if v in embedding:
                weight_matrix[k] = embedding[v]
                words_found += 1
            else:
                weight_matrix[k] = np.random.normal(size=(emb_dim, ))
                wnf.append(k)
    return weight_matrix, wnf, words_found

# Data Loader
class NMTDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    def __init__(self, input_lang, output_lang, pairs):
        """
        @param input_lang: input Lang class from which sentences need to be translated 
        @param output_lang: output Lang class where sentences are translated to
        @param pairs: list of pairs of input and output sentences
        """
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        input_sentence = self.pairs[key][0]
        input_indexes = [self.input_lang.word2index[word] if word in self.input_lang.word2index else UNK_token for word in input_sentence.split(' ')]
        input_indexes.append(EOS_token)
        input_length = len(input_indexes)

        output_sentence = self.pairs[key][1]
        output_indexes = [self.output_lang.word2index[word] if word in self.output_lang.word2index else UNK_token for word in output_sentence.split(' ')]
        output_indexes.append(EOS_token)
        output_length = len(output_indexes)
        return [input_indexes, input_length, output_indexes, output_length]

def NMTDataset_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    input_ls = []
    output_ls = []
    input_length_ls = []
    output_length_ls = []
    
    for datum in batch:
        input_length_ls.append(datum[1])
        output_length_ls.append(datum[3])
    
    # padding
    for datum in batch:
        padded_vec_input = np.pad(np.array(datum[0]), 
                                  pad_width=((0,MAX_LENGTH-datum[1])), 
                                  mode="constant", constant_values=2).tolist()
        padded_vec_output = np.pad(np.array(datum[2]), 
                                   pad_width=((0,MAX_LENGTH-datum[3])), 
                                   mode="constant", constant_values=2).tolist()
        input_ls.append(padded_vec_input)
        output_ls.append(padded_vec_output)
    return [torch.tensor(torch.from_numpy(np.array(input_ls)), device=device), 
            torch.tensor(input_length_ls, device=device), 
            torch.tensor(torch.from_numpy(np.array(output_ls)), device=device), 
            torch.tensor(output_length_ls, device=device)]

