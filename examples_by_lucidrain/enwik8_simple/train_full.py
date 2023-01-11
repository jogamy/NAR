from x_transformers import XTransformer

import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# constants

# NUM_BATCHES = int(1e5)
NUM_BATCHES = 10
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
# VALIDATE_EVERY  = 100
# GENERATE_EVERY  = 500
VALIDATE_EVERY  = 1
GENERATE_EVERY  = 1
# GENERATE_LENGTH = 1024
GENERATE_LENGTH = 10
# SEQ_LEN = 1024
SEQ_LEN = 10

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# instantiate Transformer model

model = XTransformer(
    dim = 512,
    enc_num_tokens = 256,
    enc_depth = 6,
    enc_heads = 8,
    enc_max_seq_len = 1024,
    dec_num_tokens = 256,
    dec_depth = 6,
    dec_heads = 8,
    dec_max_seq_len = 1024,
    tie_token_emb = True      # tie embeddings of encoder and decoder
)

model.cuda()

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    trX, vaX = np.split(X, [int(90e6)])
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        src = next(train_loader)
        tgt = src.clone()
        loss = model(src, tgt)
        loss.backward()

    print(f'training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        src = next(val_loader)
        tgt = src.clone()
        with torch.no_grad():
            loss = model(src, tgt)
            print(f'validation loss: {loss.item()}')

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        prime = decode_tokens(inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        # print(inp)
        # print(type(inp))

        # print(prime)
        # print(type(prime))

        # assert 1==0
        mask = torch.ones_like(src).bool()


        # def generate(self, seq_in, seq_out_start, seq_len, mask = None, attn_mask = None, **kwargs)
        sample = model.generate(seq_in=inp[0], seq_out_start=None, seq_len=GENERATE_LENGTH, mask=mask, attn_mask=mask)
        
        output_str = decode_tokens(sample)
        print(f"out: {output_str}")
