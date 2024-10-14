import torch
from torch.nn import functional as F
import math
from gpt import GPT, GPTConfig
from dataset import DataLoaderLite
import os
from hellaswag import render_example, iterate_examples

# -------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -------------------------------------------------
import tiktoken
import time
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--from_checkpoint', type=str, default=None)
parser.add_argument('--from_pretrained', type=str, default=None)

args = parser.parse_args()

if args.from_checkpoint:
    checkpoint = torch.load(args.from_checkpoint)
else:
    checkpoint = None

# detect device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(f"using device: {device}")

seed = 1337
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288 # 2^19
B = 4
T = 1024
assert total_batch_size % (B*T) == 0, 'make sure total_batch_size is divisible by B*T'
grad_accum_steps = total_batch_size // (B*T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, split='train')
val_loader = DataLoaderLite(B=B, T=T, split='val')

torch.set_float32_matmul_precision('high')

# get logits
if checkpoint:
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    print(f"loaded checkpoint from {args.from_checkpoint}")
    print(f"=> checkpoint validation loss: {checkpoint['val_loss']}")

else:  
    if args.from_pretrained:
        model = GPT.from_pretrained(args.from_pretrained)
        print(f"loaded pretrained GPT2 model {args.from_pretrained}")

    else:
        model = GPT(GPTConfig(vocab_size=50304))

model.to(device)

use_compile = False
if use_compile:
    model = torch.compile(model)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 * 5 # 19073 steps is 1 epoch

def get_lr(it):
    # 1) linear warmup 
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)

    assert 0 <=  decay_ratio <= 1
    coef = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coef * (max_lr - min_lr)

# optimize!
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
# load optimizer state dict if loading checkpoint
if checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer'])

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")

# create log file if not loading checkpoint
if not checkpoint:
    with open(log_file, 'w') as f:
        pass
    initial_step = 0

# restore current step if loading checkpoint
else:
    initial_step = checkpoint['step']
    for _ in range(initial_step):
        for micro_step in range(grad_accum_steps):
            train_loader.next_batch()

for step in range(initial_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate the model
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()

            print(f"validation loss: {val_loss_accum.item():.4f}")

            with open(log_file, 'a') as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            
            if step > 0 and (step % 5000 == 0 or last_step):
                # write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': model.state_dict(),
                    'config': model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item(),
                    'optimizer': optimizer.state_dict(),
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)

        acc_norm = num_correct_norm / num_total
        print(f"Hellaswag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        with open(log_file, 'a') as f:
            f.write(f"{step} hella {acc_norm:.4f}\n")

    # once in a while generate from model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(xgen)
                # take logits at the last position
                logits = logits[:, -1, :]
                # get probs
                probs = F.softmax(logits, dim=-1)
                # do top k sampling of 50
                # topk_probs here becomes (5, 50) and topk_indices becomes (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"generated {i}: {decoded}")

    # training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward(),
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it 
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000
    tokens_per_second = (train_loader.B * train_loader.T * grad_accum_steps) / (t1 - t0)
    print(f"step {step:4d} | loss = {loss_accum.item():.6f} | norm = {norm:.4f} | dt = {dt:.2f}ms | tok/sec: {tokens_per_second:.2f}")
    with open(log_file, "a") as f:
        f.write(f"{step} train {loss_accum.item():.6f}\n")
