from self_attn_encoder import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

batch_size = 32
SOS_token = 0
EOS_token = 1
PAD_token = 2
UNK_token = 3
MAX_LENGTH = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

"""
One-directional GRU encoder
"""
class EncoderGRU(nn.Module):
    def __init__(self, weight_mat, hidden_size, direction, layer):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.direction = direction
        self.layer = layer

        embed_mat = torch.from_numpy(weight_mat).float()
        n, embed_dim = embed_mat.shape
        self.embedding = nn.Embedding.from_pretrained(embed_mat, freeze = False)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        # get embedding of characters
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(self.direction * self.layer, batch_size, self.hidden_size, device=device)


"""
Bidirectional GRU encoder
"""
class EncoderBiGRU(nn.Module):
    def __init__(self, weight_mat, hidden_size, direction, layer, dropout_p):
        super(EncoderBiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.direction = direction
        self.layer = layer

        embed_mat = torch.from_numpy(weight_mat).float()
        n, embed_dim = embed_mat.shape
        self.embedding = nn.Embedding.from_pretrained(embed_mat, freeze = False)
        self.gru = nn.GRU(embed_dim, hidden_size, bidirectional = True, batch_first=True, dropout = dropout_p)

    def forward(self, input, hidden):        
        # get embedding of characters
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(self.direction * self.layer, batch_size, self.hidden_size, device=device)


"""
Bidirectional LSTM encoder
"""
class EncoderLSTM(nn.Module):
    def __init__(self, weight_mat, hidden_size, direction, layer, dropout_p):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.direction = direction
        self.layer = layer

        embed_mat = torch.from_numpy(weight_mat).float()
        n, embed_dim = embed_mat.shape
        self.embedding = nn.Embedding.from_pretrained(embed_mat, freeze = False)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True, dropout = dropout_p, bidirectional=True)
    
    def forward(self, input, hidden):
        # get embedding of characters
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)    
        output, hidden = self.lstm(embedded, hidden)      
        return output, hidden

    def initHidden(self, batch_size):
        return (torch.zeros(self.direction * self.layer, batch_size, self.hidden_size, device=device),
                torch.zeros(self.direction * self.layer, batch_size, self.hidden_size, device=device))



"""
One-directional GRU decoder without attention
"""
class DecoderGRU(nn.Module):
    def __init__(self, weight_mat, hidden_size, output_size):
        super(DecoderGRU, self).__init__()

        embed_mat = torch.from_numpy(weight_mat).float()
        n, embed_dim = embed_mat.shape
        self.embedding = nn.Embedding.from_pretrained(embed_mat, freeze = False)
        
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, word_input, hidden):
        # get embedding of words
        embedded = self.embedding(word_input)
        output, hidden = self.gru(embedded, hidden)
        # Final output layer
        output = output.squeeze(1) # B x N
        output = self.linear(output)
        output = self.softmax(output)
        return output, hidden


"""
Attention mechanism
"""
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(batch_size, 1, hidden_size * 4)).to(device)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(1, 2)
        
        attn_energies = self.score(hidden, encoder_outputs)
        result = F.softmax(attn_energies, dim = 2).unsqueeze(0).unsqueeze(0)
        return result
    
    def score(self, hidden, encoder_output):
        if self.method == 'general':
            energy = torch.bmm(hidden, encoder_output)
            return energy
        elif self.method == 'concat':
            seq_len = encoder_output.size()[1]
            attn_energies = torch.zeros((batch_size, 1, seq_len), device = device)  
            for i in range(seq_len):
                energy = torch.cat((hidden, (encoder_output.transpose(1, 2))[:, i, :].unsqueeze(1)), 2)
                attn_energies[:, :, i] = (torch.bmm(self.other, energy.transpose(1, 2))).squeeze(1)
            return attn_energies


"""
One-directional GRU decoder with attention
"""
class AttnDecoderGRU(nn.Module):
    def __init__(self, weight_mat, attn_model, hidden_size, output_size, n_layers, dropout_p):
        super(AttnDecoderGRU, self).__init__()
        
        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout_p)
        
        # Define layers
        embed_mat = torch.from_numpy(weight_mat).float()
        n, embed_dim = embed_mat.shape
        self.embedding = nn.Embedding.from_pretrained(embed_mat, freeze = False)
        
        self.gru = nn.GRU(hidden_size * 2 + embed_dim, hidden_size * 2, n_layers, bidirectional=False, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 4, output_size)
        
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)
    
    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(self.n_layers, batch_size, -1) # S=1 x B x N
        word_embedded = self.dropout(word_embedded)
        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)
        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output, encoder_outputs).squeeze(0).squeeze(0)
        context = attn_weights.bmm(encoder_outputs) # B x 1 x N
        # Final output layer (next word prediction) using the RNN hidden state and context vector
        output = F.log_softmax(self.out(torch.cat((rnn_output.transpose(0, 1), context), 2)), dim = 2).squeeze(1)
        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights


"""
One-directional LSTM decoder with attention
"""
class AttnDecoderLSTM(nn.Module):
    def __init__(self, weight_mat, attn_model, hidden_size, output_size, n_layers, dropout_p):
        super(AttnDecoderLSTM, self).__init__()
        
        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout_p)
        
        # Define layers
        embed_mat = torch.from_numpy(weight_mat).float()
        n, embed_dim = embed_mat.shape
        self.embedding = nn.Embedding.from_pretrained(embed_mat, freeze = False)
        
        self.lstm = nn.LSTM(hidden_size * 2 + embed_dim, hidden_size * 2, n_layers, bidirectional=False, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 4, output_size)
        
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)
    
    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(self.n_layers, batch_size, -1) # S=1 x B x N
        word_embedded = self.dropout(word_embedded)
        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context), 2)
        rnn_output, hidden = self.lstm(rnn_input, last_hidden)
        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output, encoder_outputs).squeeze(0).squeeze(0)
        context = attn_weights.bmm(encoder_outputs) # B x 1 x N
        # Final output layer (next word prediction) using the RNN hidden state and context vector
        output = F.log_softmax(self.out(torch.cat((rnn_output.transpose(0, 1), context), 2)), dim = 2).squeeze(1)
        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights

