import torch
from torch import nn
import pytorch_lightning as pl
from EncoderLSTM import EncoderLSTM
from DecoderLSTM import DecoerLSTM

class Seq2Seq(pl.LightningModule):
    def __init__(self, input_vocab=16000,output_vocab=80000, emb_dim=1000,hidden_dim=1000):
        super().__init__()
        
        self.encoder = EncoderLSTM(input_vocab=input_vocab, emb_dim=emb_dim, hidden_dim=hidden_dim)
        self.decoder = DecoderLSTM(output_vocab=output_vocab, emb_dim=emb_dim, hidden_dim=hidden_dim)

    def forward(self, src, tgt, ):
        pass


    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass