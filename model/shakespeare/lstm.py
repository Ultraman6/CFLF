import torch.nn as nn

from model.base.base_model import BaseModel


class LSTM_shakespeare(BaseModel):
    def __init__(self, mode):
        super(LSTM_shakespeare, self).__init__(mode)
        self.embeddings = nn.Embedding(num_embeddings=80, embedding_dim=8, padding_idx=0)
        self.lstm = nn.LSTM(input_size=8, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, 80)

    def forward(self, input_seq):
        embeds = self.embeddings(input_seq)
        # Note that the order of mini-batch is random so there is no hidden relationship among batches.
        # So we do not input the previous batch's hidden state,
        # leaving the first hidden state zero `self.lstm(embeds, None)`
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(embeds)
        # use the final hidden state as the next character prediction
        final_hidden_state = lstm_out[:, -1]
        output = self.fc(final_hidden_state)
        # For fed_shakespeare
        # output = self.fc(lstm_out[:,:])
        # output = torch.transpose(output, 1, 2)
        return output