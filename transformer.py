"""
transformer.py - Transformer Implementation from Scratch
Based on "Attention Is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            if mask.dim() == 4:
                scores = scores.masked_fill(~mask, -1e9)
            elif mask.dim() == 2:
                mask_expanded = mask.unsqueeze(0).unsqueeze(1)
                scores = scores.masked_fill(~mask_expanded, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.W_o(context)
        
        return output, attn_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, max_len=5000, dropout=0.1, pad_idx=0):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def create_padding_mask(self, seq, pad_idx=0):
        return (seq != pad_idx)
    
    def encode(self, src, src_mask=None):
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.positional_encoding(src_embedded)
        
        for layer in self.encoder_layers:
            src_embedded = layer(src_embedded, src_mask)
        
        return src_embedded
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.positional_encoding(tgt_embedded)
        
        for layer in self.decoder_layers:
            tgt_embedded = layer(tgt_embedded, encoder_output, src_mask, tgt_mask)
        
        return self.fc_out(tgt_embedded)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return output
    
    def translate(self, src, sos_idx, eos_idx, max_len=50):
        self.eval()
        
        src_mask = self.create_padding_mask(src).unsqueeze(1).unsqueeze(2)
        encoder_output = self.encode(src, src_mask)
        
        decoder_input = torch.tensor([[sos_idx]], device=src.device)
        
        for _ in range(max_len):
            tgt_mask = self.generate_square_subsequent_mask(decoder_input.size(1)).to(src.device)
            
            output = self.decode(decoder_input, encoder_output, src_mask, tgt_mask)
            next_token = output[:, -1].argmax(dim=-1, keepdim=True)
            
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            if next_token.item() == eos_idx:
                break
        
        return decoder_input.squeeze(0).tolist()


class TransformerTranslator(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, num_heads=4,
                 num_layers=3, d_ff=512, max_len=100, dropout=0.1):
        super().__init__()
        
        self.transformer = Transformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            pad_idx=src_vocab.token2idx['<PAD>']
        )
        
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        src_mask = self.transformer.create_padding_mask(src).unsqueeze(1).unsqueeze(2)
        
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        
        output = self.transformer(src, tgt, src_mask, tgt_mask)
        return output
    
    def translate(self, src_sentence):
        src_enc = torch.tensor([self.src_vocab.encode(src_sentence, add_eos=True)]).to(next(self.parameters()).device)
        
        return self.transformer.translate(
            src_enc,
            sos_idx=self.tgt_vocab.token2idx['<SOS>'],
            eos_idx=self.tgt_vocab.token2idx['<EOS>']
        )


def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad=1.0):
    model.train()
    total_loss = 0
    
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        src_mask = model.transformer.create_padding_mask(src).unsqueeze(1).unsqueeze(2)
        
        tgt_seq_len = tgt_input.size(1)
        tgt_mask = model.transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)
        
        optimizer.zero_grad()
        
        output = model.transformer(src, tgt_input, src_mask, tgt_mask)
        
        output = output.contiguous().view(-1, output.size(-1))
        tgt_output = tgt_output.contiguous().view(-1)
        
        loss = criterion(output, tgt_output)
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
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            src_mask = model.transformer.create_padding_mask(src).unsqueeze(1).unsqueeze(2)
            
            tgt_seq_len = tgt_input.size(1)
            tgt_mask = model.transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)
            
            output = model.transformer(src, tgt_input, src_mask, tgt_mask)
            
            output = output.contiguous().view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)
            
            loss = criterion(output, tgt_output)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


class TranslationDataset(torch.utils.data.Dataset):
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
        tgt = self.tgt_vocab.encode(self.tgt_sentences[idx], max_len=self.max_len, add_sos=True, add_eos=True)
        return torch.tensor(src), torch.tensor(tgt)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    from data_utils import Vocabulary, get_small_dataset
    
    pairs = get_small_dataset()
    src_sentences = [p[0] for p in pairs]
    tgt_sentences = [p[1] for p in pairs]
    
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.build_vocab(src_sentences)
    tgt_vocab.build_vocab(tgt_sentences)
    
    print(f"Source vocab: {len(src_vocab)}, Target vocab: {len(tgt_vocab)}\n")
    
    train_dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    model = TransformerTranslator(src_vocab, tgt_vocab, d_model=128, num_heads=4, num_layers=3, d_ff=256).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    print("Training Transformer...")
    for epoch in range(1, 101):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
            
            model.eval()
            test_sentences = ["hello", "thank you", "how are you", "goodbye"]
            
            for src in test_sentences:
                pred = model.translate(src)
                print(f"  {src} -> {tgt_vocab.decode(pred)}")
            print()


if __name__ == "__main__":
    main()
