from preprocessing import *
from models import *
from train_eval import *
import argparse
import os
import torch

batch_size = 32
SOS_token = 0
EOS_token = 1
PAD_token = 2
UNK_token = 3
MAX_LENGTH = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parser():
    parser = argparse.ArgumentParser(description="main.py")

    parser.add_argument('--path_input', type=str, default=''+os.getcwd()+'/wiki.zh.vec',
                        help='path of input language pretrained embeddings')
    parser.add_argument('--path_output', type=str, default='/'.join(os.getcwd().split('/')[:-1])+'/hw2/wiki-news-300d-1M.vec',
                        help='path of output language pretrained embeddings')
    parser.add_argument('--encoder', type=str, default='gru',
                        help='encoder type: gru,bigru,bilstm')
    parser.add_argument('--decoder', type=str, default='gru',
                        help='decoder type: gru,gruattn,lstmattn')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate in training the model.')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='hidden dimension of encoder and decoder models')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='dropout rate in training the model.')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='number of layers of the model')
    parser.add_argument('--n_directions', type=int, default=1,
                        help='number of directions of the model')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.9,
                        help='the ratio to determine whether to use teacher forcing in decoder')
    parser.add_argument('--use_attn', type=int, default=0,
                        help='whether to use attention or not')
    parser.add_argument('--print_freq', type=int, default=500,
                        help='print frequency during training')
    parser.add_argument('--epoch_num', type=int, default=20,
                        help='number of epochs to train the translation system')
    parser.add_argument('--infreq_cnt', type=int, default=1,
                        help='the number of count frequency below which the word is deleted from the vocabulary')
    parser.add_argument('--beam_k', type=int, default=2,
                        help='the value of k during beam search')
    return parser


def main():
    args = parser().parse_args()
    print("arguments: %s" %(args))


    input_path = args.path_input
    output_path = args.path_output
    enc = args.encoder
    dec = args.decoder
    LR_RATE = args.learning_rate
    hidden_size = args.hidden_size
    dropout_p = args.dropout_rate
    n_layers = args.n_layers
    n_directions = args.n_directions
    teacher_forcing_ratio = args.teacher_forcing_ratio
    use_attn = args.use_attn
    EPOCH_NUM = args.epoch_num
    PRINT_FREQ = args.print_freq
    infrequent_count = args.infreq_cnt
    beam_K = args.beam_k



    train_input_lang, train_output_lang, train_pairs = readLangs('train', 'zh', 'en', infrequent_count)
    val_input_lang, val_output_lang, val_pairs = readLangs('dev', 'zh', 'en', infrequent_count)
    test_input_lang, test_output_lang, test_pairs = readLangs('test', 'zh', 'en', infrequent_count)
    print('read_language')


    fname_zh = input_path
    fname_eng = output_path
    embedding_mat_zh = load_embedding(fname_zh)
    embedding_mat_en = load_embedding(fname_eng)
    print('load_embedding')


    chinese_wm, chin_wnf, chin_wf = create_weight(train_input_lang.index2word, embedding_mat_zh)
    english_wm, eng_wnf, eng_wf = create_weight(train_output_lang.index2word, embedding_mat_en)
    print('create_weight')


    train_dataset = NMTDataset(train_input_lang, train_output_lang, train_pairs)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size,
                                               collate_fn=NMTDataset_collate_func,
                                               shuffle=True,
                                               drop_last=True)

    val_dataset = NMTDataset(train_input_lang, train_output_lang, val_pairs)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             collate_fn=NMTDataset_collate_func,
                                             shuffle=True,
                                             drop_last=True)

    test_dataset = NMTDataset(train_input_lang, train_output_lang, test_pairs)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=batch_size,
                                             collate_fn=NMTDataset_collate_func,
                                             shuffle=True,
                                             drop_last=True)
    print('dataset')


    attn_model = 'general'
    if enc == 'gru':
        encoder = EncoderGRU(chinese_wm, hidden_size, n_directions, n_layers).to(device)
    elif enc == 'bigru':
        encoder = EncoderBiGRU(chinese_wm, hidden_size, n_directions, n_layers, dropout_p).to(device)
    elif enc == 'bilstm':
        encoder = EncoderLSTM(chinese_wm, hidden_size, n_directions, n_layers, dropout_p).to(device)

    if dec == 'gru':
        decoder = DecoderGRU(english_wm, hidden_size, train_output_lang.n_words).to(device)
    elif dec == 'gruattn':
        decoder = AttnDecoderGRU(english_wm, attn_model, hidden_size, train_output_lang.n_words, n_layers, dropout_p).to(device)
    elif dec == 'lstmattn':
        decoder = AttnDecoderLSTM(english_wm, attn_model, hidden_size, train_output_lang.n_words, n_layers, dropout_p).to(device)

    print('train model')
    trainIters(train_loader, val_loader, encoder, decoder, n_iters=EPOCH_NUM, print_every=PRINT_FREQ, learning_rate=LR_RATE, teacher_forcing_ratio=teacher_forcing_ratio, use_attn=use_attn, directions=n_directions, train_input_lang=train_input_lang, train_output_lang=train_output_lang)


if __name__ == '__main__':
    main()




