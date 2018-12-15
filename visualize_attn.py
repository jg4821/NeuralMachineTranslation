from train_eval import *
import os
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
%matplotlib inline

font_path = os.getcwd()+"/chinese_font.ttf"
prop = font_manager.FontProperties(fname=font_path)

def plotAttention(input_sentence, output_words, attentions):
    """
    Function that takes in attention and visualize the attention.
    @param - input_sentence: string the represent a list of words from source language
    @param - output_words: the gold translation in target language
    @param - attentions: a numpy array
    """
    input_sentence = input_sentence.split(' ')
    output_sentence = output_words.split(' ')

    # Set up figure with colorbar    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy()[:len(output_sentence)+1,:len(input_sentence)+2], cmap='gray')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence+
                       ['<EOS>'], rotation=90, fontproperties=prop)
    ax.set_yticklabels([''] + output_sentence+
                       ['<EOS>'])

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()



score, candidate_corpus, reference_corpus, input_corpus, attn_weights = test(encoder, decoder, test_loader, use_attn, directions, train_input_lang, train_output_lang)
for j, (input_sentence, output_words) in enumerate(zip(input_corpus, predict_corpus)):
    plotAttention(input_sentence, output_words, attn_weights[:,j,:])