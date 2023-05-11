import torch
from pytorch_metric_learning.distances import CosineSimilarity
from torch.utils.data import Dataset
from torchtext.vocab import vocab
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn import CosineSimilarity as cosine


class SiamNet(torch.nn.Module):
    def __init__(self, vocab_size, emb_size=100, hid_size=50, output_size=30, dropout=0.1):
        super().__init__()
        self.embs = torch.nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.bilstm = torch.nn.LSTM(emb_size, hid_size, bidirectional=True, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(hid_size*2, output_size)
        self.distance = CosineSimilarity()
        
    def forward(self, symbols_id):
        embs = self.embs(symbols_id) #[batch_size, sequence_len, hidden_size_in] 
        lstm_out, _ = self.bilstm(embs)  #batch_size, sequence_len, 2*hidden_size,
        pooled = torch.mean(lstm_out, 1)
        linear = self.linear(pooled)
        return linear
    
    def predict(self, symbols_id):
        embeddings = self.forward(symbols_id)
        return self.distance(embeddings)
    
class Stanza_Datatset(Dataset): 
    def __init__(self, last_words_filtered, letters_vocab, maxlen):
        super().__init__()
        self.vocab = letters_vocab
        self.words = last_words_filtered
        self.maxlen = maxlen
        self.stanzas_list = self.extract_stanzas(last_words_filtered)
    
    def extract_stanzas(self, last_words_filtered):
        all_stanzas = []
        for poem in last_words_filtered:
            stanzas_list = last_words_filtered[poem]
            all_stanzas += stanzas_list
        return all_stanzas
    
    def __len__(self):
        return len(self.stanzas_list)
        
    def __getitem__(self, item): 
        stanza = self.stanzas_list[item]
        max_word_length = max([len(word) for word in stanza])
        max_word_length = min(max_word_length, self.maxlen)
        stanza_ids = []


        for word in stanza:
            word = list(word[-self.maxlen:])
            word_id = self.vocab(word)
            word_id = [0] * (max_word_length-len(word_id)) + word_id
            stanza_ids.append(word_id)
        
        stanza_ids = torch.LongTensor(stanza_ids)    
        return stanza_ids, stanza
    
    
class Poetry_Datatset(Dataset): 
    def __init__(self, df, maxlen=10):
        super().__init__()
        self.vocab = self.create_vocab(df)
        self.words1 = list(df.word_1)
        self.words2 = list(df.word_2)
        self.maxlen = maxlen
        
    def create_vocab(self, df):
        symbol_counter=Counter()
        for word in df.word_1:
            symbol_counter.update(word)
        for word in df.word_2:
            symbol_counter.update(word) 
        letters_vocab = vocab(symbol_counter, min_freq=100, specials=['<pad>', '<unk>']) #padding index = 0, unk = 1 
        letters_vocab.set_default_index(letters_vocab["<unk>"])
        return letters_vocab
    
    def convert_sample(self, word): 
        word = list(word[-self.maxlen:])
        sample_id = self.vocab(word)
        sample_id = [0] * (self.maxlen-len(sample_id)) + sample_id
        return torch.LongTensor(sample_id)
    
    def __len__(self):
        return len(self.words1)
    
    def __getitem__(self, item):
        word1, word2 = self.words1[item], self.words2[item]
        word1 = list(word1[-self.maxlen:])
        word2 = list(word2[-self.maxlen:])
        id1 = self.vocab(word1)
        id1 = [0] * (self.maxlen-len(id1)) + id1
        id2 = self.vocab(word2)
        id2 = [0] * (self.maxlen-len(id2)) + id2
        return torch.LongTensor(id1), torch.LongTensor(id2)
    
class ReimsGen(SiamNet):
    def __init__(self, siam_param, letters_vocab, maxlen): 
        super().__init__(vocab_size=len(letters_vocab)) 
        self.load_state_dict(torch.load(siam_param))
        #self.embs.requires_grad = False
        self.embs.requires_grad_(False)
        self.letters_vocab = letters_vocab
        self.maxlen = maxlen
        self.word_embs = None
        self.words_vocab = None
        self.cos_sim = cosine(dim=1)
        
        
    def create_vocab(self, words_dataset_name):
        words_dataset = pd.read_csv(words_dataset_name)
        word_counter = Counter()
        word_counter.update(words_dataset.word_1.astype(str))
        word_counter.update(words_dataset.word_2.astype(str))
        words_vocab = vocab(word_counter)
        return words_vocab
    
    def create_W(self, words_vocab, batch_size):
        preprocessed_words = WordsDataset(words_vocab, self.letters_vocab, self.maxlen) 
 
        data_loader = DataLoader(preprocessed_words, batch_size=batch_size)
        for (letter_ids, word_ids) in data_loader:
            embeddings = self.forward(letter_ids)
            self.word_embs.weight[word_ids] = embeddings
            
    def preprocess_data(self, dataset_name, batch_size):
        words_vocab = self.create_vocab(dataset_name)
        self.words_vocab = words_vocab
        self.word_embs = torch.nn.Embedding(len(words_vocab), embedding_dim=self.linear.out_features)
        self.word_embs.requires_grad_(False)
        self.create_W(words_vocab, batch_size = batch_size)
        
    def generate_reim(self, new_word):
        new_word = new_word.lower()
        new_word_letters = list(new_word[-self.maxlen:])
        word_id = self.letters_vocab(new_word_letters)
        word_id = [0] * (self.maxlen-len(word_id)) + word_id
        word_id = torch.LongTensor([word_id]) 
        new_emb = self.forward(word_id)
        
        simillarity = self.cos_sim(new_emb, self.word_embs.weight)
        top = torch.topk(simillarity, 6).indices
        if new_word in self.words_vocab:
            top = top[1:]
        else:
            top = top[:-1]
        return self.words_vocab.lookup_tokens(list(top))
    
   
class WordsDataset(Dataset):
    def __init__(self, words_vocab, letters_vocab, maxlen):
        super().__init__()
        self.words_vocab = words_vocab
        self.letters_vocab = letters_vocab
        self.maxlen = maxlen
        
    def __len__(self):
        return len(self.words_vocab)
        
    def __getitem__(self, item): 
        word = self.words_vocab.lookup_tokens([item])[0]
        word = list(word[-self.maxlen:])
        word_id = self.letters_vocab(word)
        word_id = [0] * (self.maxlen-len(word_id)) + word_id
        return torch.LongTensor(word_id), item