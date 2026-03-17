from nltk.tokenize import word_tokenize
import re
from collections import Counter



# word tokenization

def word_tokenization(document):
    exclude = '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~'
    
    for char in exclude:
        document = document.replace(char, '')
    
    document = re.sub(r"[∗†]", "", document)
    
    tokens = word_tokenize(document.lower())
    
    return tokens


# build vocab
def build_vocab(tokens):
  vocab = {'<unk>':0}
  for token in Counter(tokens).keys():
    if token not in vocab:
      vocab[token] = len(vocab)
  return vocab

# extract sentences from data
def sentence_tokenization(document):
  input_sentences = document.split('\n')
  return input_sentences


# text to indecies
def text_to_indices(sentence, vocab):
  numerical_sentence = []

  for token in sentence:
    if token in vocab:
      numerical_sentence.append(vocab[token])
    else:
      numerical_sentence.append(vocab['<unk>'])
  return numerical_sentence


