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


def split_into_chapters(text, book_name=''):
    """Split the book into individual chapters."""
    # Multiple patterns to match different chapter formats
    patterns = [
        r'\n\n([IVX]+\.\s+[A-Z\s]+)\n\n',  # "I. A SCANDAL IN BOHEMIA"
        r'\n\nCHAPTER ([IVX]+\.?\s*[—\-\.]?\s*[^\n]*)\n\n',  # "CHAPTER I" or "CHAPTER I. Title"
        r'\n\n([IVX]+\n\n)',  # Just roman numerals
        r'\n\nChapter ([IVX]+\.?\s*[^\n]*)\n\n',  # "Chapter I" variations
    ]
    
    chapters = []
    matches = []
    
    # Try each pattern until we find chapters
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if len(matches) > 2:  # Need at least 3 chapters
            break
    
    # If no chapter markers found, split into large chunks
    if len(matches) < 3:
        # Split by double newlines or approximate equal chunks
        words = text.split()
        chunk_size = 5000  # ~5000 words per chunk
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk) > 500:
                chapters.append({
                    'title': f'Section {len(chapters) + 1}',
                    'content': chunk
                })
        
        return chapters
    
    # Process found chapters
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
    # Read all books from the dataset folder
    dataset_path = Path('dataset')
    book_files = list(dataset_path.glob('*.txt'))
    
    print(f"Found {len(book_files)} books in dataset folder")
    print("Books to process:")
    for book_file in book_files:
        print(f"  - {book_file.name}")
    
    all_chapters = []
    
    # Process each book
    for book_file in book_files:
        print(f"\nProcessing {book_file.name}...")
        
        try:
            with open(book_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Clean and split
            text = clean_text(text)
            chapters = split_into_chapters(text, book_file.stem)
            
            # Add book name to chapter metadata
            for chapter in chapters:
                chapter['book'] = book_file.stem
            
            all_chapters.extend(chapters)
            print(f"  Found {len(chapters)} chapters")
            
        except Exception as e:
            print(f"  Error processing {book_file.name}: {e}")
            continue
    
    print(f"\nTotal chapters from all books: {len(all_chapters)}")
    
    # Create training examples
    print("Creating training examples...")
    examples = create_training_examples(all_chapters, chunk_size=512, overlap=64)
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
