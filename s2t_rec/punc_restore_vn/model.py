import torch
import torch.nn as nn
from transformers import AutoModel


class BertBLSTMPunc(nn.Module):
    def __init__(
        self,
        pretrained_token,
        freeze_bert=True,
        output_size=5,
        dropout=0.0,
        bert_size=768,
        blstm_size=128,
        num_blstm_layers=2,
    ):
        super().__init__()
        self.output_size = output_size
        self.bert = AutoModel.from_pretrained(pretrained_token)

        # Freeze bert layer
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=bert_size,
            hidden_size=blstm_size,
            num_layers=num_blstm_layers,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(blstm_size * 2, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.bert(x)[0]
        y, (_, _) = self.lstm(x)

        y = self.fc(self.dropout(y))
        y = torch.reshape(y, shape=[-1, self.output_size])

        logit = self.softmax(y)

        return y, logit
