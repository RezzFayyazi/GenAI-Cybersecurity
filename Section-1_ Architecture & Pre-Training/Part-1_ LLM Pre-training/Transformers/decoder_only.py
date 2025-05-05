import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
from transformers import GPT2Tokenizer
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



class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, target_mask):
        attn_output = self.self_attn(x, x, x, target_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(DecoderOnlyTransformer, self).__init__()
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)

        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, tgt):
        # Generate the causal mask (target mask) only
        seq_length = tgt.size(1)
        tgt_mask = torch.tril(torch.ones(seq_length, seq_length)).unsqueeze(0).unsqueeze(1)  # Causal mask
        return tgt_mask.to(tgt.device)  # Ensure mask is on the same device as `tgt`
    
    def forward(self, tgt):
        tgt_mask = self.generate_mask(tgt)
        tgt_embedded = self.dropout(self.pos_encoding(self.decoder_embedding(tgt)))

        
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, tgt_mask)
        
        output = self.fc(dec_output)
        return output


# TRAINING THE TRANSFORMER MDOEL
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Model hyperparameters
tgt_vocab_size = tokenizer.vocab_size  # GPT-2 tokenizer vocab size
d_model = 2048
num_heads = 16
num_layers = 8
d_ff = 2048
max_seq_length = 100
dropout = 0.1

# Instantiate the Decoder-Only Transformer model
transformer = DecoderOnlyTransformer(tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

texts = ["Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks",
         "Hackers are individuals who attempt to gain unauthorized access to computer systems",
         "Vulnerabilities are weaknesses in software that can be exploited by attackers",
         "Phishing is a type of social engineering attack that aims to trick people into providing sensitive information",
         "Ransomware is a type of malware that encrypts a victim's files and demands payment for their release",
         "Zero-day vulnerabilities are previously unknown software flaws that can be exploited by attackers",
         ]

# Tokenize the texts
tokenized_data = [
    tokenizer.encode(text, max_length=max_seq_length, truncation=True, padding='max_length')
    for text in texts
]

# Convert tokenized data to tensors
dataset = torch.tensor(tokenized_data)

# Split into training and validation sets
training_data = dataset[:4]
test_data = dataset[4:]

# Add dummy labels (required by DataLoader and loss function)
train_dataset = TensorDataset(training_data, training_data)
val_dataset = TensorDataset(test_data, test_data)

# Data loaders
batch_size = 2  # Ensure batch size matches dataset size for testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # Use padding token index
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Training loop
for epoch in range(5):  # Reduce epochs for testing
    transformer.train()
    total_loss = 0

    for batch in train_loader:
        tgt_data, _ = batch  # `_` is a placeholder for dummy labels

        optimizer.zero_grad()
    
        # Forward pass: the model takes in the sequence except for the last token and predicts the next token
        output = transformer(tgt_data[:, :-1])
        
        # Reshape outputs and targets to match vocabulary dimension
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    # Print loss for each epoch
    print(f"Epoch: {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

# Evaluation of the Transformer model
transformer.eval()
val_loss = 0
examples = []

with torch.no_grad():
    for val_batch in val_loader:
        val_tgt_data, _ = val_batch  # `_` is a placeholder for dummy labels
        
        # Forward pass on validation data
        val_output = transformer(val_tgt_data[:, :-1])
    
        vloss = criterion(
            val_output.contiguous().view(-1, tgt_vocab_size),
            val_tgt_data[:, 1:].contiguous().view(-1)
        )
        val_loss += vloss.item()

        # Collect some examples for inspection
        for i in range(min(len(val_tgt_data), 3)):  # Save a few examples
            decoded_input = tokenizer.decode(val_tgt_data[i].tolist(), skip_special_tokens=True)
            decoded_output = tokenizer.decode(
                torch.argmax(val_output[i], dim=-1).tolist(),
                skip_special_tokens=True
            )
            examples.append((decoded_input, decoded_output))

    # Print validation loss
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
    
    # Print example predictions
    for i, (inp, out) in enumerate(examples):
        print(f"Example {i + 1}:\nInput: {inp}\nOutput: {out}\n")