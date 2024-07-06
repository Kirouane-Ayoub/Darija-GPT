import pandas as pd
import torch
from transformers import AutoTokenizer

import settings

splits = {"v1": "data/v1-00000-of-00001.parquet"}
df = pd.read_parquet("hf://datasets/ayoubkirouane/Algerian-Darija/" + splits["v1"])
text = df["Text"].to_list()

with open("train_data/cleaned_data.txt", "w") as f:
    f.write("\n".join(text))

# Load the pre-trained GPT-2 tokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")


# Define a generator function to read text data in batches from a file
def text_iterator(file_path, batch_size=1000):
    # Open the file in read mode with UTF-8 encoding
    with open(file_path, encoding="utf-8") as file:
        batch = []  # Initialize an empty list to store the batch of lines
        # Iterate over each line in the file
        for line in file:
            batch.append(
                line.strip()
            )  # Strip any leading/trailing whitespace and add the line to the batch
            # If the batch size is reached, yield the batch and reset the batch list
            if len(batch) == batch_size:
                yield batch
                batch = []
        # Yield any remaining lines in the batch that didn't reach the batch size
        if batch:
            yield batch


# Train a new tokenizer on the Algerian Darija text data using the text_iterator function
darija_tokenizer = tokenizer.train_new_from_iterator(
    text_iterator(file_path=settings.DATASET_PATH, batch_size=1000), vocab_size=25000
)

# Save the newly trained Darija tokenizer to a directory
darija_tokenizer.save_pretrained("darija_tokenizer")


# The DataLoader class defines a data loader for the GPT-2 model
class DataLoader:
    # The __init__ method initializes the data loader with the batch size (B) and the sequence length (T)
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # Open the data file and read the text
        with open("data.txt") as f:
            text = f.read()

        # Tokenize the text using the darija_tokenizer
        tokens = darija_tokenizer.encode(text)

        # Convert the tokens to a PyTorch tensor
        self.tokens = torch.tensor(tokens)

        # Initialize the current position to 0
        self.current_position = 0

    # The next_batch method returns the next batch of data
    def next_batch(self):
        # Get the batch size (B) and the sequence length (T)
        B, T = self.B, self.T

        # Get the next batch of tokens from the current position
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]

        # Split the batch of tokens into input (x) and target (y) sequences
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        # Update the current position
        self.current_position += T * B

        # If the current position is beyond the end of the data, reset it to 0
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        # Return the input and target sequences
        return x, y
