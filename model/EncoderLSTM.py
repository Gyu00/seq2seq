import torch.nn as nn

class EncoderLSTM(nn.Module):
    def __init__(self, input_vocab, emb_dim, hidden_dim, n_layers=4, p=0.1):
        super().__init__()

        self.encoder_embedding = nn.Embedding(num_embeddings=input_vocab, embedding_dim=emb_dim)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=n_layers, dropout =p )
        self.dropout = nn.Dropout(p=p)

    def forward(self, x,input_lengths):

        x = self.encoder_embedding(x)
        x= self.dropout(x)

        output, (h_n, c_n) = self.lstm(x)
        return output, h_n, c_n