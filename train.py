import torch
import torch.nn as nn

# training sequences
def training_sequences(input_numerical_sentences):
  training_sequence = []
  for sentence in input_numerical_sentences:
    for i in range(1, len(sentence)):
      training_sequences.append(sentence[:i+1])
  return training_sequence

# padding

def padding(training_sequence):
  len_list = []
  for sequence in training_sequence:
    len_list.append(len(sequence))
  
  padded_training_sequence = []
  for sequence in training_sequence:
    padded_training_sequence.append([0]*(max(len_list) - len(sequence)) + sequence)

  # converting the 2D list into tensor
  padded_training_sequence = torch.tensor(padded_training_sequence, dtype=torch.long)
  return padded_training_sequence


class LSTMModel(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    self.embedding = nn.Embedding(1565, 100)
    self.lstm = nn.LSTM(100, 150, batch_first=True)
    self.fc = nn.Linear(150, vocab_size)

  def forward(self, x):
    embedded = self.embedding(x)
    intermediate_hidden_states, (final_hidden_state, final_cell_state) = self.lstm(embedded)
    output = self.fc(final_hidden_state.squeeze(0))
    return output
