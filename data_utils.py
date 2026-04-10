"""
data_utils.py - Data preprocessing for Seq2Seq translation
"""

import numpy as np
import re


class Vocabulary:
    def __init__(self):
        self.token2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2token = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.n_tokens = 4
    
    def build_vocab(self, sentences, min_freq=1):
        token_freq = {}
        for sent in sentences:
            tokens = self._tokenize(sent)
            for token in tokens:
                token_freq[token] = token_freq.get(token, 0) + 1
        
        for token, freq in token_freq.items():
            if freq >= min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
                self.n_tokens += 1
    
    def _tokenize(self, text):
        return re.findall(r'\w+|[^\s\w]', text.lower())
    
    def encode(self, sentence, max_len=None, add_sos=False, add_eos=True):
        tokens = self._tokenize(sentence)
        indices = [self.token2idx.get(t, self.token2idx['<UNK>']) for t in tokens]
        
        if add_sos:
            indices = [self.token2idx['<SOS>']] + indices
        if add_eos:
            indices.append(self.token2idx['<EOS>'])
        
        if max_len is not None:
            if len(indices) < max_len:
                indices += [self.token2idx['<PAD>']] * (max_len - len(indices))
            else:
                indices = indices[:max_len]
        
        return indices
    
    def decode(self, indices, skip_special=True):
        tokens = []
        for idx in indices:
            token = self.idx2token.get(idx, '<UNK>')
            if skip_special and token in ['<PAD>', '<SOS>', '<EOS>']:
                continue
            tokens.append(token)
        return ' '.join(tokens)
    
    def __len__(self):
        return self.n_tokens


def load_parallel_corpus(source_file, target_file, max_samples=None):
    """Load parallel corpus for translation."""
    source_sentences = []
    target_sentences = []
    
    with open(source_file, 'r', encoding='utf-8') as sf, \
         open(target_file, 'r', encoding='utf-8') as tf:
        for src, tgt in zip(sf, tf):
            source_sentences.append(src.strip())
            target_sentences.append(tgt.strip())
            
            if max_samples and len(source_sentences) >= max_samples:
                break
    
    return source_sentences, target_sentences


def create_batches(source_seqs, target_seqs, batch_size, src_vocab, tgt_vocab, max_len=50):
    """Create batches for training."""
    X, y = [], []
    
    for src, tgt in zip(source_seqs, target_seqs):
        src_enc = src_vocab.encode(src, max_len=max_len, add_sos=False, add_eos=True)
        tgt_enc = tgt_vocab.encode(tgt, max_len=max_len, add_sos=True, add_eos=True)
        
        X.append(src_enc)
        y.append(tgt_enc)
    
    X = np.array(X).T
    y = np.array(y).T
    
    n_batches = len(source_sentences) // batch_size
    X = X[:, :n_batches * batch_size]
    y = y[:, :n_batches * batch_size]
    
    X = X.reshape(src_vocab.n_tokens, -1, batch_size)
    y = y.reshape(tgt_vocab.n_tokens, -1, batch_size)
    
    return X, y


def get_small_dataset():
    """Get a small toy dataset for testing."""
    pairs = [
        ("hello", "bonjour"),
        ("goodbye", "au revoir"),
        ("thank you", "merci"),
        ("please", "s'il vous plait"),
        ("yes", "oui"),
        ("no", "non"),
        ("good morning", "bonjour"),
        ("good night", "bonne nuit"),
        ("how are you", "comment allez vous"),
        ("i love you", "je t'aime"),
        ("what is your name", "comment vous appelez vous"),
        ("my name is", "je m'appelle"),
        ("where is the bathroom", "ou est la salle de bain"),
        ("i don't understand", "je ne comprends pas"),
        ("speak english", "parlez anglais"),
        ("how much", "combien"),
        ("water", "eau"),
        ("food", "nourriture"),
        ("help", "aide"),
        ("stop", "arret"),
    ]
    return pairs


def prepare_data(pairs, src_vocab=None, tgt_vocab=None, max_len=50):
    """Prepare data from translation pairs."""
    if src_vocab is None:
        src_vocab = Vocabulary()
    if tgt_vocab is None:
        tgt_vocab = Vocabulary()
    
    src_sentences = [p[0] for p in pairs]
    tgt_sentences = [p[1] for p in pairs]
    
    src_vocab.build_vocab(src_sentences)
    tgt_vocab.build_vocab(tgt_sentences)
    
    return src_sentences, tgt_sentences, src_vocab, tgt_vocab


class DataIterator:
    """Iterator for training data."""
    
    def __init__(self, source_seqs, target_seqs, src_vocab, tgt_vocab, 
                 batch_size=32, max_len=50):
        self.source_seqs = source_seqs
        self.target_seqs = target_seqs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.batch_size = batch_size
        self.max_len = max_len
        self.n_samples = len(source_seqs)
        self.indices = np.arange(self.n_samples)
        np.random.shuffle(self.indices)
        self.pos = 0
    
    def __iter__(self):
        self.pos = 0
        np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        if self.pos >= self.n_samples:
            raise StopIteration
        
        batch_indices = self.indices[self.pos:self.pos + self.batch_size]
        
        X_batch = []
        y_batch = []
        
        for idx in batch_indices:
            src_enc = np.array(self.src_vocab.encode(
                self.source_seqs[idx], max_len=self.max_len, add_eos=True
            ))
            tgt_enc = np.array(self.tgt_vocab.encode(
                self.target_seqs[idx], max_len=self.max_len, add_sos=True, add_eos=True
            ))
            X_batch.append(src_enc)
            y_batch.append(tgt_enc)
        
        X_batch = np.array(X_batch).T.reshape(-1, self.max_len + 1, len(batch_indices))
        y_batch = np.array(y_batch).T.reshape(-1, self.max_len + 2, len(batch_indices))
        
        self.pos += self.batch_size
        return X_batch, y_batch
    
    def __len__(self):
        return self.n_samples // self.batch_size


if __name__ == "__main__":
    vocab = Vocabulary()
    test_sentence = "hello world"
    encoded = vocab.encode(test_sentence)
    decoded = vocab.decode(encoded)
    print(f"Test: '{test_sentence}' -> {encoded} -> '{decoded}'")
