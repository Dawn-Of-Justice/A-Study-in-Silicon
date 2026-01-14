"""
Prepare the Sherlock Holmes dataset for fine-tuning.
This script splits the book into manageable chunks for training.
"""

import json
import re
from pathlib import Path


def clean_text(text):
    """Remove Project Gutenberg header and footer."""
    # Find the start of the actual content
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    
    start_idx = text.find(start_marker)
    if start_idx != -1:
        # Find the end of the start marker line
        start_idx = text.find('\n', start_idx) + 1
        text = text[start_idx:]
    
    end_idx = text.find(end_marker)
    if end_idx != -1:
        text = text[:end_idx]
    
    return text.strip()


def split_into_chapters(text):
    """Split the book into individual chapters."""
    # Pattern to match chapter headings like "I. A SCANDAL IN BOHEMIA"
    chapter_pattern = r'\n\n([IVX]+\.\s+[A-Z\s]+)\n\n'
    
    chapters = []
    matches = list(re.finditer(chapter_pattern, text))
    
    for i, match in enumerate(matches):
        start = match.start()
        # Get chapter title
        title = match.group(1).strip()
        
        # Get chapter content (until next chapter or end)
        if i < len(matches) - 1:
            end = matches[i + 1].start()
        else:
            end = len(text)
        
        content = text[start:end].strip()
        
        if len(content) > 100:  # Only include substantial chapters
            chapters.append({
                'title': title,
                'content': content
            })
    
    return chapters


def create_training_examples(chapters, chunk_size=1024, overlap=128):
    """
    Create training examples from chapters using a sliding window approach.
    Each example is a chunk of text that the model will learn to continue.
    """
    examples = []
    
    for chapter in chapters:
        text = chapter['content']
        words = text.split()
        
        # Create overlapping chunks
        i = 0
        while i < len(words):
            chunk = ' '.join(words[i:i + chunk_size])
            
            if len(chunk) > 200:  # Minimum chunk size
                examples.append({
                    'text': chunk,
                    'chapter': chapter['title']
                })
            
            i += chunk_size - overlap
    
    return examples


def main():
    # Read the book
    book_path = Path('dataset/the-adventures-of-sherlock-holmes.txt')
    with open(book_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Clean and split
    print("Cleaning text...")
    text = clean_text(text)
    
    print("Splitting into chapters...")
    chapters = split_into_chapters(text)
    print(f"Found {len(chapters)} chapters")
    
    # Create training examples
    print("Creating training examples...")
    examples = create_training_examples(chapters, chunk_size=512, overlap=64)
    print(f"Created {len(examples)} training examples")
    
    # Split into train/validation (90/10)
    split_idx = int(len(examples) * 0.9)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    # Save as JSONL files
    output_dir = Path('processed_data')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'train.jsonl', 'w', encoding='utf-8') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')
    
    with open(output_dir / 'validation.jsonl', 'w', encoding='utf-8') as f:
        for example in val_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"\nDataset created successfully!")
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(f"Files saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
