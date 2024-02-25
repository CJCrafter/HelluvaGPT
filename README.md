# HelluvaGPT
HelluvaGPT is a simple LLM (Large Language Model) inspired by Andrej Kaparthy's videos
on [tokenization]() and building a [LLM from scratch]().
The example model provided is trained on a small dataset of [Helluva Boss]()
transcripts.

# Usage

### Running the project
You need to be running Python 3.7 or later. You can install the dependencies using pip:
```shell
pip install -r requirements.txt
```

To actually run the code, you can use the `main.py` file:
```shell
python main.py
```

### Using your own dataset
You can replace the files in `./llm_training` with any `.txt` files you want to train your model on.


# Technical Details

### Tokenization
The tokenizer is based on GPT-4's tokenizer, which is a byte pair encoding tokenizer.
This works by first splitting the text into chunks of words, numbers, and whitespace,
then encoding the text into bytes, and finally encoding the bytes into tokens.

If you replace the data in `./llm_training` with your own data, you will need to retrain the tokenizer.
You can do this by deleting the `./helluvatokenizer.model` and `./helluvatokenizer.vocab` files,
then running `./main.py` (It will detect that there is no tokenizer, and train a new one).

### Special Tokens
We only added 1 special token, `[<endoftext>]`. This is used to deliminate between the scripts 
in the training data. Ideally, the trained model will understand that `[<endoftext>]` means that
it should ignore any context from the previous script.

### Context Length
Context length is variable, but is set to `256` by default. On my notebook 3060, I was able
to train the model with a context length of 256 in about 3 minutes. Increasing the context
length will increase the time it takes to train the model, but it will also improve the model's
performance. 

### Hyperparameters
The hyperparameters are declared at the top of `./main.py`. 

```python
# Tokenizer Hyperparameters
vocab_size = 2000          # the number of tokens in the vocabulary
name = "helluvatokenizer"  # the name of the tokenizer

# Language Model Hyperparameters
batch_size = 32            # how many sequences to process in parallel
block_size = 256           # the context window size
learning_rate = 1e-3       # how quickly to learn
steps = 10000              # how many training steps to take
eval_steps = 100           # steps between evaluations during training
n_embed = 32               # the size of the token embeddings
head_size = 8              # the number of layers in the multiheadattention model
num_heads = 4              # the number of heads in the multiheadattention model
device = "cuda" if torch.cuda.is_available() else "cpu"
```