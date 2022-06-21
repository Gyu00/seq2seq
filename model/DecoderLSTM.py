import torch.nn as nn

class DecoderLSTM(nn.Module):
    def __init__(self, output_vocab,emb_dim, hidden_dim,n_layers=4, p=0.1):
        super().__init__()

        self.decoder_embedding = nn.Embedding(num_embeddings=output_vocab,embedding_dim=emb_dim)
        
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=n_layers, dropout =p )
        
        self.fc = nn.Linear(hidden_dim, output_vocab)

        self.dropout = nn.Dropout(p=p)


    def forward(self, x, h_n, c_n):
        x = x.unsqueeze(0)

        x = self.decoder_embedding(x)
        x = self.dropout(x)

        output, (h_n, c_n) = self.lstm(x, (h_n, c_n))

        pred = self.fc(output.squeeze(0))

        return pred, h_n, c_n