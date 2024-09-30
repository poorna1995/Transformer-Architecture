

from torch.utils.data import Dataset, DataLoader 
import tiktoken
import torch
import torch.nn as nn


class GPTDatasetV1(Dataset):
    def __init__(self, text_data, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(text_data, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for start in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[start:start + max_length]
            target_chunk = token_ids[start + 1: start + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def create_gpt_dataloader(text_data, batch_size=4, max_length=256,
                           stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(text_data, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
            GELU(),
            nn.Linear(4 * config["emb_dim"], config["emb_dim"]),
        )

    def forward(self, input_tensor):
        return self.layers(input_tensor)


class LayerNormalization(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, input_tensor):
        mean = input_tensor.mean(dim=-1, keepdim=True)
        var = input_tensor.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (input_tensor - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        return 0.5 * input_tensor * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (input_tensor + 0.044715 * torch.pow(input_tensor, 3))
        ))


# Unique generate text function
def generate_next_tokens(model, input_indices, max_new_tokens, context_size):
    # input_indices is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        input_context = input_indices[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(input_context)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the index of the vocab entry with the highest logits value
        next_index = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        input_indices = torch.cat((input_indices, next_index), dim=1)  # (batch, n_tokens+1)

    return input_indices






