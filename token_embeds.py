import tiktoken
import torch
enc = tiktoken.get_encoding("gpt2")
max_bytes = 16
vocab_size = 50304  # padded
table = torch.zeros(vocab_size, max_bytes, dtype=torch.int32)
for i in range(enc.n_vocab):
    b = list(enc.decode_single_token_raw(i))[:max_bytes]
    table[i, :len(b)] = torch.tensor(b, dtype=torch.int32)
torch.save(table, "data/spelling_table.pt")