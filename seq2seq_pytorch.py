"""
seq2seq_pytorch.py - Seq2Seq Model using PyTorch
Production-ready implementation with GPU support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class Encoder(nn.Module):
    """LSTM Encoder that produces context vector."""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class Decoder(nn.Module):
    """LSTM Decoder that generates output sequence."""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predictions = self.fc(outputs)
        return predictions, hidden, cell


class Seq2SeqModel(nn.Module):
    """Complete Seq2Seq model with Encoder-Decoder."""
    
    def __init__(self, input_vocab_size, output_vocab_size,
                 embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = Encoder(input_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(output_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        self.output_vocab_size = output_vocab_size
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        
        outputs = torch.zeros(batch_size, tgt_len, self.output_vocab_size, device=src.device)
        
        hidden, cell = self.encoder(src)
        
        decoder_input = tgt[:, 0:1]
        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t] = output.squeeze(1)
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(2)
            decoder_input = tgt[:, t:t+1] if teacher_force else top1
        
        return outputs
    
    def translate(self, src, sos_idx, eos_idx, max_len=50):
        self.eval()
        with torch.no_grad():
            hidden, cell = self.encoder(src)
            decoder_input = torch.tensor([[sos_idx]], device=src.device)
            
            translations = []
            for _ in range(max_len):
                output, hidden, cell = self.decoder(decoder_input, hidden, cell)
                top1 = output.argmax(2)
                token = top1.item()
                
                if token == eos_idx:
                    break
                translations.append(token)
                decoder_input = top1
            
            return translations


class Seq2SeqAttention(nn.Module):
    """Seq2Seq with Bahdanau Attention."""
    
    def __init__(self, input_vocab_size, output_vocab_size,
                 embed_dim=256, hidden_dim=512, num_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_vocab_size = output_vocab_size
        
        self.encoder_embedding = nn.Embedding(input_vocab_size, embed_dim, padding_idx=0)
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        
        self.decoder_embedding = nn.Embedding(output_vocab_size, embed_dim, padding_idx=0)
        self.decoder_lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, num_layers, batch_first=True)
        
        self.W_a = nn.Linear(hidden_dim, hidden_dim)
        self.U_a = nn.Linear(hidden_dim, hidden_dim)
        self.v_a = nn.Linear(hidden_dim, 1)
        
        self.fc = nn.Linear(hidden_dim, output_vocab_size)
    
    def attention(self, s_t, encoder_outputs, mask=None):
        s_t_expanded = s_t.unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)
        scores = self.v_a(torch.tanh(self.W_a(s_t_expanded) + self.U_a(encoder_outputs)))
        scores = scores.squeeze(2)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        
        return context, attn_weights
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        
        encoder_outputs, (hidden, cell) = self.encoder_lstm(self.encoder_embedding(src))
        
        outputs = []
        attn_weights_list = []
        
        decoder_input = tgt[:, 0:1]
        for t in range(1, tgt_len):
            s_t, (hidden, cell) = self.decoder_lstm(
                self.decoder_embedding(decoder_input), (hidden, cell)
            )
            s_t = s_t.squeeze(1)
            
            context, attn_weights = self.attention(s_t, encoder_outputs)
            attn_weights_list.append(attn_weights)
            
            combined = torch.cat([s_t.unsqueeze(1), context.unsqueeze(1)], dim=2)
            output = self.fc(combined)
            outputs.append(output)
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(2)
            decoder_input = tgt[:, t:t+1] if teacher_force else top1
        
        return torch.cat(outputs, dim=1)


class TranslationDataset(Dataset):
    """Dataset for translation pairs."""
    
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=50):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src = self.src_vocab.encode(self.src_sentences[idx], max_len=self.max_len, add_eos=True)
        tgt = self.tgt_vocab.encode(self.tgt_sentences[idx], max_len=self.max_len, 
                                     add_sos=True, add_eos=True)
        return torch.tensor(src), torch.tensor(tgt)


def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad=1.0):
    model.train()
    total_loss = 0
    
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        
        optimizer.zero_grad()
        
        output = model(src, tgt, teacher_forcing_ratio=0.5)
        
        output = output[:, 1:].contiguous().view(-1, output.size(-1))
        tgt = tgt[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, tgt)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            output = model(src, tgt, teacher_forcing_ratio=0)
            
            output = output[:, 1:].contiguous().view(-1, output.size(-1))
            tgt = tgt[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, tgt)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    from data_utils import Vocabulary, get_small_dataset, prepare_data
    
    pairs = get_small_dataset()
    src_sentences = [p[0] for p in pairs]
    tgt_sentences = [p[1] for p in pairs]
    
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.build_vocab(src_sentences)
    tgt_vocab.build_vocab(tgt_sentences)
    
    print(f"Source vocab: {len(src_vocab)}, Target vocab: {len(tgt_vocab)}\n")
    
    train_dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    model = Seq2SeqModel(
        input_vocab_size=len(src_vocab),
        output_vocab_size=len(tgt_vocab),
        embed_dim=128,
        hidden_dim=256,
        num_layers=2
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training...")
    for epoch in range(1, 101):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
            
            model.eval()
            test_sentences = ["hello", "thank you", "how are you"]
            sos_idx = tgt_vocab.token2idx['<SOS>']
            eos_idx = tgt_vocab.token2idx['<EOS>']
            
            for src in test_sentences:
                src_enc = torch.tensor([src_vocab.encode(src, add_eos=True)]).to(device)
                pred = model.translate(src_enc, sos_idx, eos_idx)
                print(f"  {src} -> {tgt_vocab.decode(pred)}")
            print()


if __name__ == "__main__":
    main()
