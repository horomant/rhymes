import torch
from siam_net import SiamNet
from torchtext.vocab import vocab
from collections import Counter
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn import CosineSimilarity as cosine
    
class ReimsGen(SiamNet):
    def __init__(self, siam_param, letters_vocab, maxlen): # размер embs не передаётся на прямую, но достаётся из linar слоя Сиама
        super().__init__(vocab_size=len(letters_vocab)) #init родительского класса
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
        word_counter.update(words_dataset.Word_1)
        word_counter.update(words_dataset.Word_2)
        words_vocab = vocab(word_counter)
        return words_vocab
    
    def create_W(self, words_vocab, batch_size):
        preprocessed_words = WordsDataset(words_vocab, self.letters_vocab, self.maxlen) # тут лежат слова, переведённый в id по буквам
 
        data_loader = DataLoader(preprocessed_words, batch_size=batch_size)
        for (letter_ids, word_ids) in data_loader:
            embeddings = self.forward(letter_ids)
            self.word_embs.weight[word_ids] = embeddings
            
    def preprocess_data(self, dataset_name, batch_size):
        words_vocab = self.create_vocab(dataset_name)
        self.words_vocab = words_vocab
        self.word_embs = torch.nn.Embedding(len(words_vocab), embedding_dim=self.linear.out_features)
        #self.word_embs.requires_grad_ = False
        self.word_embs.requires_grad_(False)
        self.create_W(words_vocab, batch_size = batch_size)
        
    def generate_reim(self, new_word):
        new_word = new_word.lower()
        new_word_letters = list(new_word[-self.maxlen:])
        word_id = self.letters_vocab(new_word_letters)
        word_id = [0] * (self.maxlen-len(word_id)) + word_id
        word_id = torch.LongTensor([word_id]) # оборачиваем word_id ещё в список, чтобы добавить измерение батча
        new_emb = self.forward(word_id)
        
        simillarity = self.cos_sim(new_emb, self.word_embs.weight)
        #reim = torch.argmax(distances)
        top = torch.topk(simillarity, 6).indices
        if new_word in self.words_vocab:
            top = top[1:]
        else:
            top = top[:-1]
        
        print(top)
        #print(self.words_vocab.lookup_tokens(list(top)))
        return self.words_vocab.lookup_tokens(list(top))
    
class WordsDataset(Dataset):
    def __init__(self, words_vocab, letters_vocab, maxlen):
        super().__init__()
        self.words_vocab = words_vocab
        self.letters_vocab = letters_vocab
        self.maxlen = maxlen
        
    def __len__(self):
        return len(self.words_vocab)
        
    def __getitem__(self, item): #item = индекс куска данных, которые хотим достать. Тут item - id слова
        word = self.words_vocab.lookup_tokens([item])[0]
        word = list(word[-self.maxlen:])
        word_id = self.letters_vocab(word)
        word_id = [0] * (self.maxlen-len(word_id)) + word_id
        return torch.LongTensor(word_id), item