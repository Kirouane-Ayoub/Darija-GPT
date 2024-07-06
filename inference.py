import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# Define the function to perform inference
def generate_text(prompt, model_path, max_length=30, num_return_sequences=5):
    # Load the trained tokenizer
    darija_tokenizer = AutoTokenizer.from_pretrained("darija_tokenizer")

    # Initialize the model and load the saved weights
    model = AutoModelForCausalLM.from_config(darija_tokenizer.config)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # Tokenize the input prompt
    tokens = darija_tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)

    # Add a batch dimension and repeat for num_return_sequences
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = tokens.to(device)
    model.to(device)

    # Generate text
    while x.size(1) < max_length:
        with torch.no_grad():
            outputs = model(x)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)

    # Decode and print the generated sequences
    generated_texts = []
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = darija_tokenizer.decode(tokens)
        generated_texts.append(decoded)
    return generated_texts


# Set up argparse
def main():
    parser = argparse.ArgumentParser(description="Generate text using Darija-GPT")
    parser.add_argument(
        "--prompt", type=str, required=True, help="Input prompt text in Algerian Darija"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the saved model file"
    )
    parser.add_argument(
        "--max_length", type=int, default=30, help="Maximum length of generated text"
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=5,
        help="Number of sequences to generate",
    )
    args = parser.parse_args()

    generated_texts = generate_text(
        args.prompt, args.model_path, args.max_length, args.num_return_sequences
    )
    for i, text in enumerate(generated_texts):
        print(f"Generated Sequence {i+1}: {text}")


if __name__ == "__main__":
    main()
