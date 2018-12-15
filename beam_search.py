import os
import numpy as np
from io import open
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
import math
import time
import random
import pickle as pkl
import sacrebleu
from sacrebleu import corpus_bleu
from sacrebleu import raw_corpus_bleu

batch_size = 32
SOS_token = 0
EOS_token = 1
PAD_token = 2
UNK_token = 3
MAX_LENGTH = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



##### GRU Attn #####
def beam_search(decoder, decoder_hidden, encoder_output, decoder_context, beam_k, max_length):
    de_hidden = None
    de_context = None
    words_mat = None  # in fact, a list structure
    prob_mat = None
    hidden_mat = []
    hidden_index = []  # keep track of which hidden state to input for next state
    context_mat = []
    
    for di in range(max_length):
        # initialize the beam for search
        if words_mat == None:
            decoder_input = torch.tensor([[SOS_token]]*batch_size, device=device)
            decoder_output, de_context, de_hidden, attn_weights= decoder(decoder_input, decoder_context, decoder_hidden, encoder_output)
            de_context = de_context.transpose(0, 1)
#             decoder_output, de_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output)
            topv, topi = decoder_output.topk(k=beam_k)  # B*k

            prob_mat = topv.cpu().numpy()  # B*k
            if di == 0:
                words_mat = np.expand_dims(topi.cpu().numpy(),2).tolist()  # B*k*1
        else:
            temp_sum_prob = np.zeros((batch_size, beam_k**2))  # B*k^2
            temp_all_topi = np.zeros((batch_size, beam_k**2),dtype=int)  # B*k^2
            # get previous k word tokens for each sentence in a batch
            previous_words_mat = []
            for sent in words_mat:
                # sent: 1*k*1
                temp = []
                for li in sent:
                    temp.append(li[-1])
                previous_words_mat.append(temp)
            previous_words_mat = np.asarray(previous_words_mat)  # B*k
            
            # do beam search on k top words
            for word_ind in range(beam_k):
                potential_word = previous_words_mat[:,word_ind]
                decoder_input = torch.from_numpy(potential_word).to(device).unsqueeze(1)  # B*1
                if len(hidden_index) != 0:
                    de_hidden = torch.from_numpy(hidden_mat[hidden_index[word_ind]]).to(device)
                    de_context = torch.from_numpy(context_mat[hidden_index[word_ind]]).to(device)
                
                decoder_output, context, hidden, attn_weights= decoder(decoder_input, de_context, de_hidden, encoder_output)
#                 decoder_output, hidden, attn_weights = decoder(decoder_input, de_hidden, encoder_output)
                
                topv, topi = decoder_output.topk(k=beam_k)  # B*k
                temp_sum_prob[:,(word_ind*beam_k):(word_ind+1)*beam_k] = np.expand_dims(prob_mat[:,word_ind],1) + topv.cpu().numpy()
                temp_all_topi[:,(word_ind*beam_k):(word_ind+1)*beam_k] = topi.cpu().numpy()
                
                hidden_mat.append(hidden.cpu().numpy())
                context_mat.append(context.transpose(0,1).cpu().numpy())
            hidden_mat = hidden_mat[-beam_k:]
            context_mat = context_mat[-beam_k:]
            
            # sort k^2 results by probability descending and keep the top k
            prob_topk = -np.sort(-temp_sum_prob,1)[:,:beam_k]  # B*k
            ind_topk = np.argsort(-temp_sum_prob,1)[:,:beam_k]  # B*k
            # update current sum of probability matrix
            prob_mat = prob_topk
            # update words matrix
            # process one sentence at a time
            for i in range(batch_size):
                for j in range(beam_k):
                    cur_wordIndex = ind_topk[i,j]
                    cur_word = temp_all_topi[i,cur_wordIndex]
                    q, r = divmod(cur_wordIndex,beam_k)  # q: previous word index
                    # get previous word corresponding list
                    li = words_mat[i][q].copy()
                    li.append(cur_word)
                    words_mat[i].append(li.copy())
                    if len(hidden_index) == beam_k:
                        hidden_index[j] = q
                    else:
                        hidden_index.append(q)
                for num in range(beam_k):
                    words_mat[i].pop(0)
    # process last word, find max probability prediction
    ind_max = np.argsort(prob_mat,1)[:,-1]  # B*1
    decoded_words = []
    for i in range(batch_size):
        decoded_words.append(words_mat[i][ind_max[i]])
    return np.asarray(decoded_words)

##### beam search evaluate function #####
def bs_evaluate(encoder, decoder, input, beam_k, max_length):
    # process input sentence
    with torch.no_grad():

        encoder_hidden = encoder.initHidden(batch_size)
        encoder_output, encoder_hidden = encoder(input, encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]]*batch_size, device=device)
        # decode the context vector
        decoder_hidden = encoder_hidden # decoder starts from the last encoding sentence

        encoder_outputs = encoder_output
        decoder_context = torch.zeros((1, batch_size, decoder.hidden_size * 2), device = device)  # 1*B*2h

        decoded_words = beam_search(decoder,decoder_hidden,encoder_outputs,decoder_context,beam_k,max_length)
        return decoded_words

##### beam search test function #####
def bs_test(encoder, decoder, data_loader, beam_k, max_length, train_input_lang, train_output_lang):
    count = 0
    
    candidate_corpus = []
    reference_corpus = []

    for i, (input, input_len, target, target_len) in enumerate(data_loader):
        decoded_words = bs_evaluate(encoder, decoder, input, beam_k, max_length)
        candidate_sentences = []
        for ind in range(decoded_words.shape[0]):
            sent_words = []
            for token in decoded_words[ind]:
                if token != PAD_token and token != EOS_token:
                    sent_words.append(train_output_lang.index2word[token])
                else:
                    break
            sent_words = ' '.join(sent_words)
            # print the first sentence in the first batch to peek the translation result
            if count == 0:
                print('predict: '+sent_words)
                count += 1
            candidate_sentences.append(sent_words)
        candidate_corpus.extend(candidate_sentences)

        reference_sentences = []
        for sent in target:
            sent_words = []
            for token in sent:
                if token.item() != EOS_token:
                    sent_words.append(train_output_lang.index2word[token.item()])
                else:
                    break
            sent_words = ' '.join(sent_words)
            if count == 1:
                print('target: '+sent_words)
                count += 1
            reference_sentences.append(sent_words)
        reference_corpus.extend(reference_sentences)
    
    score = corpus_bleu(candidate_corpus, [reference_corpus], smooth='exp', smooth_floor=0.0, force=False).score
    return score

