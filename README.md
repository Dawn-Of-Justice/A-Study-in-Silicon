# Sherlock Holmes Fine-Tuning Project

Fine-tune Google's Gemma model to write in the style of Sherlock Holmes.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Hugging Face Authentication:**
   - Create an account at https://huggingface.co
   - Get your access token from https://huggingface.co/settings/tokens
   - Accept the Gemma model license at https://huggingface.co/google/gemma-2b
   - Login via CLI:
     ```bash
     huggingface-cli login
     ```

## Workflow

### Step 1: Prepare the Dataset
```bash
python prepare_dataset.py
```
This will:
- Clean the Sherlock Holmes book
- Split it into chapters
- Create training chunks with overlap
- Save train/validation splits to `processed_data/`

### Step 2: Fine-tune the Model
```bash
python finetune_gemma.py
```
This will:
- Load the Gemma-2B model (or 7B if you change the config)
- Apply LoRA for efficient fine-tuning
- Train on your Sherlock Holmes dataset
- Save the fine-tuned model to `sherlock-gemma-model/`

**Note:** Training requires a GPU. For CPU-only or limited RAM:
- Use Gemma-2B instead of 7B
- Reduce batch size in the script
- Consider using Google Colab or other cloud GPUs

### Step 3: Generate Text
```bash
python generate_sherlock.py
```
This will:
- Load your fine-tuned model
- Let you enter prompts
- Generate Sherlock-style continuations

## Example Prompts

Try these prompts with your fine-tuned model:
- "To Sherlock Holmes, the case was"
- "I had not seen my friend Holmes for"
- "The mystery of the"
- "Watson, come quickly! I have discovered"

## Configuration Options

### In `prepare_dataset.py`:
- `chunk_size`: Size of text chunks (default: 512 tokens)
- `overlap`: Overlap between chunks (default: 64 tokens)

### In `finetune_gemma.py`:
- `MODEL_NAME`: "google/gemma-2b" or "google/gemma-7b"
- `BATCH_SIZE`: Adjust based on your GPU memory
- `NUM_EPOCHS`: Number of training epochs (default: 3)
- `LEARNING_RATE`: Learning rate (default: 2e-4)

### In `generate_sherlock.py`:
- `max_length`: Maximum length of generated text
- `temperature`: Creativity (0.7-1.0, higher = more creative)
- `top_p`: Nucleus sampling parameter

## Adding More Books

To add more Sherlock Holmes books:
1. Add .txt files to the `dataset/` folder
2. Modify `prepare_dataset.py` to read multiple files
3. Re-run the preparation and training steps

## Hardware Requirements

**Minimum:**
- GPU: 8GB VRAM (for Gemma-2B with 8-bit quantization)
- RAM: 16GB
- Storage: 20GB free space

**Recommended:**
- GPU: 16GB+ VRAM (for Gemma-7B)
- RAM: 32GB
- Storage: 50GB free space

## Troubleshooting

**CUDA out of memory:**
- Reduce `BATCH_SIZE` in `finetune_gemma.py`
- Increase `GRADIENT_ACCUMULATION_STEPS`
- Use Gemma-2B instead of 7B

**Import errors:**
- Ensure all packages are installed: `pip install -r requirements.txt`
- Update transformers: `pip install --upgrade transformers`

**Generation quality issues:**
- Train for more epochs
- Add more books to the dataset
- Adjust temperature and top_p parameters
