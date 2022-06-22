import torch
from torch import nn
import pytorch_lightning as pl
import random
from EncoderLSTM import EncoderLSTM
from DecoderLSTM import DecoderLSTM

class Seq2Seq(pl.LightningModule):
    def __init__(self, input_vocab=16000,output_vocab=80000, emb_dim=1000,hidden_dim=1000):
        super().__init__()
        self.output_vocab=output_vocab
        self.encoder = EncoderLSTM(input_vocab=input_vocab, emb_dim=emb_dim, hidden_dim=hidden_dim)
        self.decoder = DecoderLSTM(output_vocab=output_vocab, emb_dim=emb_dim, hidden_dim=hidden_dim)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index = 0)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):

        encoder_out, h_n, c_n = self.encoder(src)
        inputs = tgt[0,:]
        outputs = torch.zeros(tgt.shape[0], tgt.shape[1], self.output_vocab).to(self.device)
        for t in range(1, tgt.shape[0]):
            output, hidden, cell = self.decoder(inputs,h_n, c_n)
            outputs[t]=output

            inputs = tgt[t] if (random.random()<teacher_forcing_ratio) else output.argmax(1)

        return outputs


    def training_step(self, batch, batch_idx):
        src, tgt = batch
        output = self(src, tgt)
        loss = self.loss_fn( output[1:].view(-1, output.shape[-1]), tgt[1:].view(-1) )
        self.log('train_loss', loss)

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        output = self(src, tgt)
        loss = self.loss_fn( output[1:].view(-1, output.shape[-1], tgt[1:].view(-1)) )
        self.log('train_loss', loss)

    def configure_optimizers(self):
        pass