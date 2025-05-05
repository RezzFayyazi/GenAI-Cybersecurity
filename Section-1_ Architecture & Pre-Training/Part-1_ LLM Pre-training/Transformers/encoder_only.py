import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask==0, -1e9)
        
        attn_probs = torch.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_probs, V)
        return output
    
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, Q, K, V, mask=None):

        Q = self.split_heads(self.W_Q(Q))
        K = self.split_heads(self.W_K(K))
        V = self.split_heads(self.W_V(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        output = self.W_O(self.combine_heads(attn_output))

        return output
    

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedforward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class EncoderOnlyTransformer(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(EncoderOnlyTransformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, src_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src):
        # Generate a mask for the source (padding mask)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # Mask padding tokens
        return src_mask.to(src.device)  # Ensure mask is on the same device as `src`
    
    def forward(self, src):
        # Generate the source mask
        src_mask = self.generate_mask(src)
        
        # Apply embedding and positional encoding
        src_embedded = self.dropout(self.pos_encoding(self.encoder_embedding(src)))

        # Pass through each encoder layer
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
    
        # Apply final linear layer to map to vocabulary size
        output = self.fc(enc_output)
        return output


# --- Data Preparation ---

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

texts = [
        "Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks",
         "Hackers are individuals who attempt to gain unauthorized access to computer systems",
         "Vulnerabilities are weaknesses in software that can be exploited by attackers",
]

max_seq_length = 64
tokenized_data = [
    tokenizer.encode(text, max_length=max_seq_length, truncation=True, padding='max_length')
    for text in texts
]

# Mask random tokens for MLM task
masked_data = []
for seq in tokenized_data:
    masked_seq = seq[:]
    for i in range(len(masked_seq)):
        if torch.rand(1).item() < 0.15 and seq[i] not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            masked_seq[i] = tokenizer.mask_token_id
    masked_data.append(masked_seq)

# Prepare datasets
dataset = torch.tensor(masked_data)
target_dataset = torch.tensor(tokenized_data)
train_dataset = TensorDataset(dataset, target_dataset)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# --- Model and Training Setup ---

src_vocab_size = tokenizer.vocab_size
d_model = 256
num_heads = 4
num_layers = 2
d_ff = 512
dropout = 0.1

transformer = EncoderOnlyTransformer(src_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001)

# --- Validation Data ---
val_texts = [
         "Phishing is a type of social engineering attack that aims to trick people into providing sensitive information",
         "Ransomware is a type of malware that encrypts a victim's files and demands payment for their release",
         "Zero-day vulnerabilities are previously unknown software flaws that can be exploited by attackers",
]

val_tokenized_data = [
    tokenizer.encode(text, max_length=max_seq_length, truncation=True, padding='max_length')
    for text in val_texts
]

# Apply the same random masking to validation data
val_masked_data = []
for seq in val_tokenized_data:
    masked_seq = seq[:]
    for i in range(len(masked_seq)):
        if torch.rand(1).item() < 0.15 and seq[i] not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            masked_seq[i] = tokenizer.mask_token_id
    val_masked_data.append(masked_seq)

val_dataset = torch.tensor(val_masked_data)
val_target_dataset = torch.tensor(val_tokenized_data)
val_loader = DataLoader(TensorDataset(val_dataset, val_target_dataset), batch_size=2, shuffle=False)


# --- Training ---

for epoch in range(5):
    transformer.train()
    total_loss = 0

    for batch in train_loader:
        src_data, tgt_data = batch

        optimizer.zero_grad()
        output = transformer(src_data)

        loss = criterion(output.view(-1, src_vocab_size), tgt_data.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    # Validation
    transformer.eval()
    val_loss = 0
    examples = []

    with torch.no_grad():
        for val_batch in val_loader:
            val_src_data, val_tgt_data = val_batch
            val_output = transformer(val_src_data)

            # Compute validation loss
            vloss = criterion(val_output.view(-1, src_vocab_size), val_tgt_data.view(-1))
            val_loss += vloss.item()

            # Collect examples for inspection
            for i in range(min(len(val_src_data), 3)):  # Save a few examples
                decoded_input = tokenizer.decode(val_src_data[i].tolist(), skip_special_tokens=True)
                decoded_output = tokenizer.decode(
                    torch.argmax(val_output[i], dim=-1).tolist(),
                    skip_special_tokens=True
                )
                examples.append((decoded_input, decoded_output))

    print(f"Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader):.4f}")

    # Display some example predictions
    print("\nExamples:")
    for i, (inp, out) in enumerate(examples):
        print(f"Example {i + 1}:\nInput (masked): {inp}\nOutput (prediction): {out}\n")