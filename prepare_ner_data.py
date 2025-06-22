import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import os

# --- Configuration ---
CONLL_FILE_PATH = 'labeled_telegram_product_price_location.txt'
OUTPUT_DIR = 'data/' # Directory to save processed spaCy data
TRAIN_DATA_PATH = os.path.join(OUTPUT_DIR, 'train.spacy')
DEV_DATA_PATH = os.path.join(OUTPUT_DIR, 'dev.spacy') # For development/validation
TRAIN_SPLIT_RATIO = 0.8 # 80% for training, 20% for development

# --- Helper function to parse CoNLL-like format ---
def parse_conll(file_path):
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        tokens = []
        labels = []
        for line in f:
            line = line.strip()
            if not line: # Empty line indicates end of sentence
                if tokens: # Process the collected tokens and labels
                    text = " ".join(tokens)
                    entities = []
                    current_idx = 0 # Track current position in text for character spans

                    for i, (token, label) in enumerate(zip(tokens, labels)):
                        # Find the token in the full text string from the current search position
                        start_char = text.find(token, current_idx)
                        if start_char == -1:
                            print(f"Warning: Could not find token '{token}' from index {current_idx} in text: '{text}'")
                            current_idx = text.find(" ", current_idx) + 1 if text.find(" ", current_idx) != -1 else len(text) + 1 # Try to advance to next space
                            continue
                        end_char = start_char + len(token)

                        # Handle B- and I- tags (B-LABEL, I-LABEL)
                        if label.startswith('B-'):
                            entities.append([start_char, end_char, label[2:]]) # [start, end, LABEL]
                        elif label.startswith('I-'):
                            # Extend the last entity if it's the same type and follows immediately
                            # Check for immediate adjacency (allowing for a single space between tokens)
                            if entities and entities[-1][2] == label[2:] and entities[-1][1] == start_char - 1:
                                entities[-1][1] = end_char # Extend end_char of previous span
                            else:
                                # This handles cases where an I-tag appears without a preceding B-tag
                                # or if the label type changes. Treat it as a new entity for robustness.
                                entities.append([start_char, end_char, label[2:]])
                        # 'O' (Outside) tags are ignored as they are not entities

                        current_idx = end_char + 1 # Move past the current token + the space after it

                    # Add the processed example
                    examples.append((text, {"entities": entities}))
                # Reset for the next sentence
                tokens = []
                labels = []
            elif line.startswith('#'): # Skip comment lines like '# Message ID: 7403'
                continue
            else:
                # CRITICAL CHANGE: Split by the FIRST space, not tab
                # This handles "TOKEN LABEL" format
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    token = parts[0]
                    label = parts[1]
                    tokens.append(token)
                    labels.append(label)
                else:
                    # This warning helps identify malformed lines in your data
                    print(f"Warning: Skipping malformed line (expected 'TOKEN LABEL'): '{line}'")
        
        # After the loop, add the last sentence if the file doesn't end with a newline
        if tokens:
            text = " ".join(tokens)
            entities = []
            current_idx = 0
            for i, (token, label) in enumerate(zip(tokens, labels)):
                start_char = text.find(token, current_idx)
                if start_char == -1:
                    current_idx = text.find(" ", current_idx) + 1 if text.find(" ", current_idx) != -1 else len(text) + 1
                    continue
                end_char = start_char + len(token)

                if label.startswith('B-'):
                    entities.append([start_char, end_char, label[2:]])
                elif label.startswith('I-'):
                    if entities and entities[-1][2] == label[2:] and entities[-1][1] == start_char - 1:
                        entities[-1][1] = end_char
                    else:
                        entities.append([start_char, end_char, label[2:]])
                current_idx = end_char + 1
            examples.append((text, {"entities": entities}))
    return examples

# --- Main processing ---
def create_spacy_docbin(data, output_path):
    """
    Converts a list of (text, entities) tuples into a spaCy DocBin and saves it.
    """
    # Load a blank model specifically for creating Doc objects without processing pipelines
    # This is more robust for training data preparation
    nlp = spacy.blank("en") # Use a blank English model

    db = DocBin()
    
    # Handle empty data case gracefully
    if not data:
        print(f"Warning: No data provided for {output_path}. Creating an empty DocBin.")
        db.to_disk(output_path)
        return

    docs = []
    # Use tqdm for progress bar
    for text, annot in tqdm(data, desc=f"Processing to DocBin for {output_path}"):
        doc = nlp.make_doc(text) # Create Doc object
        ents = []
        for start, end, label in annot["entities"]:
            # Use alignment_mode="contract" to handle slight tokenization differences
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                # This often happens if the character spans don't perfectly align with tokens
                # For training, it's crucial to debug these or adjust tokenization/span creation.
                print(f"Skipping entity due to alignment error: '{text[start:end]}' ({label}) in text: '{text}'")
                continue
            ents.append(span)
        doc.ents = ents
        docs.append(doc)

    # CRITICAL FIX: Add each Doc object individually to DocBin
    for doc in docs:
        db.add(doc)

    db.to_disk(output_path)
    print(f"Successfully created {output_path} with {len(docs)} documents.")


if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Parsing CoNLL file: {CONLL_FILE_PATH}")
    all_examples = parse_conll(CONLL_FILE_PATH)
    print(f"Parsed {len(all_examples)} examples from {CONLL_FILE_PATH}.")

    # Split data into training and development sets
    num_train = int(len(all_examples) * TRAIN_SPLIT_RATIO)
    train_examples = all_examples[:num_train]
    dev_examples = all_examples[num_train:]

    print(f"Creating training DocBin at {TRAIN_DATA_PATH} ({len(train_examples)} examples)")
    create_spacy_docbin(train_examples, TRAIN_DATA_PATH)

    print(f"Creating development DocBin at {DEV_DATA_PATH} ({len(dev_examples)} examples)")
    create_spacy_docbin(dev_examples, DEV_DATA_PATH)

    print("Data preparation complete. Ready for spaCy training configuration.")