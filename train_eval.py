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


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

# Training
def train(input, target, input_len, target_len, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length, teach_forcing_ratio, use_attn, directions):
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0
    ppl = 0

    encoder_hidden = encoder.initHidden(batch_size)
    encoder_output, encoder_hidden = encoder(input, encoder_hidden)

    if use_attn:
        decoder_context = torch.zeros((1, batch_size, decoder.hidden_size * 2), device = device)
        decoder_input = torch.tensor([[SOS_token]]*batch_size, device = device)
        encoder_outputs = torch.cat([encoder_output[:, :, :encoder.hidden_size], encoder_output[:, :, encoder.hidden_size:]], dim = 2)
        decoder_hidden = torch.cat([encoder_hidden[0, :, :] , encoder_hidden[1, :, :]], dim = 1).unsqueeze(0)
    
        use_teacher_forcing = True if random.random() < teach_forcing_ratio else False
        if use_teacher_forcing:
            for di in range(max_length):
                decoder_output, decoder_context, decoder_hidden, attn_weights = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
                decoder_context = decoder_context.transpose(0, 1)
                loss += criterion(decoder_output, target[:,di])
                decoder_input = target[:,di].unsqueeze(1)  # Teacher forcing (batch_size, 1)
        else:
            for di in range(max_length):
                decoder_output, decoder_context, decoder_hidden, attn_weights= decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach().unsqueeze(1)  # detach from history as input
                decoder_context = decoder_context.transpose(0, 1)
                loss += criterion(decoder_output, target[:,di])
                ni = topi[0][0]
                if ni == EOS_token:
                    break
    else:
        decoder_input = torch.tensor([[SOS_token]]*batch_size, device = device)
        if directions == 2:
            encoder_outputs = torch.cat([encoder_output[:, :, :encoder.hidden_size], encoder_output[:, :, encoder.hidden_size:]], dim = 2)
            decoder_hidden = torch.cat([encoder_hidden[0, :, :] , encoder_hidden[1, :, :]], dim = 1).unsqueeze(0)
        else:
            decoder_hidden = encoder_hidden
    
        use_teacher_forcing = True if random.random() < teach_forcing_ratio else False
        if use_teacher_forcing:
            for di in range(max_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                loss += criterion(decoder_output, target[:,di])
                decoder_input = target[:,di].unsqueeze(1)  # Teacher forcing (batch_size, 1)
        else:
            for di in range(max_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach().unsqueeze(1)  # detach from history as input
                loss += criterion(decoder_output, target[:,di])
                ni = topi[0][0]
                if ni == EOS_token:
                    break
                
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    ppl = 10 ** (loss.item() / float(max_length))
    return loss.item() / float(max_length), ppl

def trainIters(loader, test_loader, encoder, decoder, n_iters, print_every, learning_rate, teacher_forcing_ratio, use_attn, directions, train_input_lang, train_output_lang):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()
    
    best_bleu = None
    save_path = os.getcwd() + '/saved_model/test_py_file.pt'

    train_loss_hist = []
    bleu_hist = []
    
    for iter in range(1, n_iters + 1):
        print_loss_total = 0  # Reset every print_every
        print_total_perplexity = 0
        for i, (input, input_len, target, target_len) in enumerate(loader):
            loss, ppl = train(input, target, input_len, target_len, encoder, decoder, 
                         encoder_optimizer, decoder_optimizer, criterion, 
                         max_length=MAX_LENGTH, teach_forcing_ratio=teacher_forcing_ratio, use_attn=use_attn, directions=directions)
            print_loss_total += loss
            print_total_perplexity += ppl  
            
            if (i + 1) % print_every == 0:
                current_bleu, candidate_corpus, reference_corpus, input_corpus, attn_weights = test(encoder, decoder, test_loader, use_attn, directions, train_input_lang, train_output_lang)
                if not best_bleu or current_bleu > best_bleu:
                    torch.save({
                                'epoch': iter,
                                'encoder_state_dict': encoder.state_dict(),
                                'decoder_state_dict': decoder.state_dict(),
                                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                                'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                                'train_loss': loss,
                                'train_ppl': ppl,
                                'best_BLEU': best_bleu
                                }, save_path)
                    best_bleu = current_bleu
                
                print_loss_avg = print_loss_total / print_every
                print_ppl_avg = print_total_perplexity / print_every 
                print_loss_total = 0
                print_total_perplexity = 0
                train_loss_hist.append(print_loss_avg)
                bleu_hist.append(current_bleu)
                
                print('%s (Epoch: %d %d%%) | Train Loss: %.4f | Best Bleu: %.4f | Current Blue: %.4f | Perplexity: %.4f' 
                      % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg, best_bleu, current_bleu, print_ppl_avg))
                with open('test_py_file.txt', 'a') as f:
                    f.write('%s (Epoch: %d %d%%) | Train Loss: %.4f | Best Bleu: %.4f | Current Blue: %.4f | Perplexity: %.4f \n' 
                      % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg, best_bleu, current_bleu, print_ppl_avg))

        #encoder_optimizer, learning_rate = exp_lr_scheduler(encoder_optimizer, iter, lr_rate = learning_rate, lr_decay_epoch=5)
        #decoder_optimizer, learning_rate = exp_lr_scheduler(decoder_optimizer, iter, lr_rate = learning_rate, lr_decay_epoch=5)
    pkl.dump(train_loss_list, open('test_py_file_loss.p', 'wb'))
    pkl.dump(bleu_hist, open('test_py_file_bleu.p', 'wb'))

# used to schedule a learning rate decay
def exp_lr_scheduler(optimizer, epoch, lr_rate =0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0:
        lr_rate = lr_rate * (0.1**(epoch // lr_decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_rate
        return optimizer, lr_rate
    return optimizer, lr_rate


# Function that generate translation
def evaluate(encoder, decoder, input, max_length, use_attn, directions):
    # process input sentence
    with torch.no_grad():

        encoder_hidden = encoder.initHidden(batch_size)
        encoder_output, encoder_hidden = encoder(input, encoder_hidden)

        decoded_words = []
        attention_list = []
        attn_concat = None
        if use_attn:
            decoder_context = torch.zeros((1, batch_size, decoder.hidden_size * 2), device = device)
            decoder_input = torch.tensor([[SOS_token]]*batch_size, device = device)
            encoder_outputs = torch.cat([encoder_output[:, :, :encoder.hidden_size], encoder_output[:, :, encoder.hidden_size:]], dim = 2)
            decoder_hidden = torch.cat([encoder_hidden[0, :, :] , encoder_hidden[1, :, :]], dim = 1).unsqueeze(0)
            for di in range(max_length):
                decoder_output, decoder_context, decoder_hidden, attn_weights= decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoded_words.append(topi.cpu().numpy())
                decoder_input = topi.squeeze().detach().unsqueeze(1)  # detach from history as input
                decoder_context = decoder_context.transpose(0, 1)
                attention_list.append(attn_weights)
            attn_concat = torch.cat(attention_list, dim=1)
        else:
            decoder_input = torch.tensor([[SOS_token]]*batch_size, device = device)
            if directions == 2:
                encoder_outputs = torch.cat([encoder_output[:, :, :encoder.hidden_size], encoder_output[:, :, encoder.hidden_size:]], dim = 2)
                decoder_hidden = torch.cat([encoder_hidden[0, :, :] , encoder_hidden[1, :, :]], dim = 1).unsqueeze(0)
            else:
                decoder_hidden = encoder_hidden
            for di in range(max_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoded_words.append(topi.cpu().numpy())
                decoder_input = topi.squeeze().detach().unsqueeze(1)  # detach from history as input

        return np.asarray(decoded_words).T, attn_concat


def test(encoder, decoder, data_loader, use_attn, directions, train_input_lang, train_output_lang):
    count = 0
    
    input_corpus = []
    candidate_corpus = []
    reference_corpus = []

    for i, (input, input_len, target, target_len) in enumerate(data_loader):
        decoded_words, attn_weights = evaluate(encoder, decoder, input, max_length=MAX_LENGTH, use_attn=use_attn, directions=directions)
        for i in range(input.size()[0]): 
            input_words = []
            input_sent = []
            for token in input[i]: 
                token = token.item()
                if token != PAD_token and token != EOS_token:
                    input_words.append(train_input_lang.index2word[token])
                else:
                    break
            input_sent = ' '.join(input_words)
            # print the first sentence in the first batch to peek the translation result
            if count == 0:
                print('input: '+input_sent)
                count += 1
            input_corpus.extend([input_sent])
            
        candidate_sentences = []
        for ind in range(decoded_words.shape[1]):
            sent_words = []
            for token in decoded_words[0][ind]:
                if token != PAD_token and token != EOS_token:
                    sent_words.append(train_output_lang.index2word[token])
                else:
                    break
            sent_words = ' '.join(sent_words)
            if count == 1:
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
            if count == 2:
                print('target: '+sent_words)
                count += 1
            reference_sentences.append(sent_words)
        reference_corpus.extend(reference_sentences)
    
    score = corpus_bleu(candidate_corpus, [reference_corpus], smooth='exp', smooth_floor=0.0, force=False).score
    return score, candidate_corpus, reference_corpus, input_corpus, attn_weights


