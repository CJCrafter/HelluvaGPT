from llm import BigramLanguageModel
from tokenizer.regex import RegexTokenizer
import torch
import os
import time

# Tokenizer Hyperparameters
vocab_size = 2000
name = "helluvatokenizer"

# Language Model Hyperparameters
batch_size = 32  # how many sequences to process in parallel
block_size = 256  # the context window size
learning_rate = 1e-3
steps = 10000
eval_steps = 100
n_embed = 32
head_size = 8
num_heads = 4
device = "cuda" if torch.cuda.is_available() else "cpu"


# Inform the user if CUDA is available
if device == "cuda":
    print("-" * 80)
    print("CUDA is available!")
    print("Found GPU: " + torch.cuda.get_device_name(0))
    print("-" * 80)
else:
    print("!" * 80)
    print("CUDA is not available, cannot use GPU. Defaulting to CPU")
    print("!" * 80)


# text is for training the tokenizer, text_delimited is for training the language model
# (the tokenizer should not be trained on special tokens, like "[<endoftext>]")
text = ""
text_delimited = ""
delimiter = "[<endoftext>]"
for file in os.listdir("llm_training"):
    if file.endswith(".txt"):
        with open(os.path.join("llm_training", file), "r") as f:
            text += f.read() + "\n"
            text_delimited += delimiter + f.read()

# If the tokenizer has already been trained, load it, otherwise train it
tokenizer = RegexTokenizer()
if not os.path.exists(name + ".model"):
    tokenizer.train(text, vocab_size, verbose=True)
    tokenizer.register_special_tokens({"<endoftext>": vocab_size + 0})
    tokenizer.save(name)
else:
    tokenizer.load("helluvatokenizer.model")

# Tokenize our training data, and split it into a training and validation set.
# We will train on the training set, and use the validation set to score our model.
encoded = tokenizer.encode(text)
data = torch.tensor(encoded, dtype=torch.long)
n = int(0.90 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    llm.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_steps)
        for k in range(eval_steps):
            xb, yb = get_batch(split)
            _, loss = llm(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    llm.train()
    return out


# Initialize the language model and the optimizer
llm = BigramLanguageModel(vocab_size, n_embed, block_size, head_size, num_heads)
llm = llm.to(device)
optimizer = torch.optim.AdamW(llm.parameters(), lr=learning_rate)

start_time = time.time()
for steps in range(steps):

    # Every few steps, we should evaluate our model to see how it's improving
    if steps % eval_steps == 0:
        losses = estimate_loss()
        dt = time.time() - start_time
        print(f"[{dt:.1f}s] Step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")
    logits, loss = llm(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate new text using our model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
text = tokenizer.decode(llm.generate(context, 1080)[0].tolist())
print(text)