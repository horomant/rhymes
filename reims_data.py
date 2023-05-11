import torch
from torch.utils.data import Dataset
from collections import Counter
from torchtext.vocab import vocab


class Stanza_Datatset(Dataset): #создаёт тензор из id последних слов одной строФы
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
        
    def __getitem__(self, item): #item = индекс куска данных, которые хотим достать. Тут item - индекс строфы
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
    
    
class Poetry_Datatset(Dataset): # делает позитивные пары слов и переводит эти пары в id
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
    
    def convert_sample(self, word): #превращает 1 слово в id
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