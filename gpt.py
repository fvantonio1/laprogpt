import torch 
from src.layers import GPTLanguageModel
import tiktoken

# set hyperparameters
batch_size = 32 # number of independent sequences in a batch
block_size = 256 # maximum context lenght
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
n_head = 6
n_layer = 6
dropout = 0.2
# ------------
torch.manual_seed(42)

# read data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# unique caracteres
# chars = sorted(list(set(text)))
# vocab_size = len(chars)
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

# create mapping caracteres to integers and vice versa
# stoi = {ch: i for i, ch in enumerate(chars)}
# itos = {i: ch for i, ch in enumerate(chars)}
# encode = lambda s: [stoi[c] for c in s] # take a string, output a list of integers
# decode = lambda l: ''.join([itos[i] for i in l]) # take a list of integers, output a string

# train and test splits
#data = torch.tensor(encode(text), dtype=torch.long)
data = torch.tensor(enc.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# create model
model = GPTLanguageModel(n_head, n_embed, dropout, block_size,
                         vocab_size, n_layer, device)
m = model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a pytorch optimizer
opt = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# train loop
for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f'iter {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')

    # get batch
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = m(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(enc.decode(m.generate(context, 1000)[0].tolist()))
open('more2.txt', 'w').write(enc.decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
