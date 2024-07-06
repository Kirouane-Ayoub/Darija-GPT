# Training and Inference for Darija-GPT

## Overview

This project involves training and using a **GPT-2** model fine-tuned on Algerian Darija, a spoken dialect with limited written resources. The model, named **Darija-GPT**, aims to generate coherent text in **🇩🇿Algerian Darija**.


## Dataset: Algerian-Darija
### Introducing the Algerian-Darija Dataset

I have compiled a new dataset named **[Algerian-Darija](https://huggingface.co/datasets/ayoubkirouane/Algerian-Darija)**. This dataset contains text in **🇩🇿Algerian Darija**, collected from a variety of sources, including existing datasets on Hugging Face, web scraping, and YouTube comments and transcript APIs. The **v1** split consists of more than **170,000 rows** of split and partially cleaned text.

### Sources : 
**The text data was gathered from:**

- **Hugging Face Datasets:** Pre-existing datasets relevant to Algerian Darija.
- **Web Scraping:** Content from various online sources, ensuring a diverse range of topics and contexts.
- **YouTube API:** Transcriptions from Algerian Darija videos and comments on YouTube, capturing conversational and colloquial speech.

**Note:** Some text data from the YouTube Transcript API may contain imperfections due to limitations in speech-to-text technology for Algerian Darija. Additionally, the dataset still requires further cleaning to improve its quality for more advanced NLP tasks.

This dataset forms the foundation for our project, Darija-GPT, which aims to train a GPT-2 model specifically on Algerian Darija, harnessing this unique linguistic resource to build a robust and effective language model.

## Training

##### Prerequisites

1. **Training Data**: Ensure your raw text data is placed in the `train_data` folder. The data should be pre-processed and cleaned for optimal training results.
2. **Dependencies**: Install necessary Python packages, primarily from the `transformers` and `torch` libraries.

##### Training Script

The `train.py` script initiates the training process. Here are key parameters and configurations:

- **Device**: The code is set up to utilize GPU if available, falling back to CPU otherwise.
  ```python
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  ```

- **Model Initialization**: By default, the script initializes the GPT-2 large model (`gpt2-large`) with 774 million parameters, specified by the configuration:
  ```python
  n_layer=36, n_head=20, n_embd=1280
  ```
  You can change the model by setting different values in the `GPTConfig` or using a different model name through command-line arguments.

- **Hyperparameters**: The following hyperparameters can be adjusted to refine the training process:
  ```python
  max_lr = 6e-4
  min_lr = max_lr * 0.1 # 10%
  warmup_steps = 10
  max_steps = 4000
  save_interval = 500  # Save the model every 500 steps
  ```

##### Running the Training Script

Use the following command to start training:
```bash
python train.py --model_name gpt2-large
```
You can choose other model variants such as `gpt2`, `gpt2-medium`, `gpt2-xl` by specifying them with the `--model_name` argument.

## Inference

##### Inference Script

The `inference.py` script generates text using the trained Darija-GPT model. The key parameters include:

- **Prompt**: The initial text prompt in Algerian Darija.
- **Model Path**: Path to the saved model file (e.g., `model_final.pt`).
- **Max Length**: Maximum length of the generated text.
- **Number of Sequences**: Number of sequences to generate.

##### Running the Inference Script

Use the following command to generate text:
```bash
python inference.py --prompt "وحد نهار" --model_path "model_final.pt" --max_length 30 --num_return_sequences 5
```

This will generate and print multiple sequences of text based on the provided prompt.

### Final Notes

- **Training Data**: Ensure your training data is well-prepared and placed in the `train_data` folder. By default, the script will use the Algerian-Darija dataset. If you have your own dataset, make sure to place it in the `train_data` folder and update the path in the `settings.py` file.
- **Model Customization**: Customize the model configuration and hyperparameters according to your specific needs.
- **Device Compatibility**: The code is designed to utilize available GPUs for faster training and inference.

By following this guide, you can train and utilize the **Darija-GPT** model effectively for generating text in **🇩🇿Algerian Darija**.