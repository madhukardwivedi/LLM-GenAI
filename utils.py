import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def adjust_ranges(tag_entry):
    if pd.isna(tag_entry):
        return tag_entry
    updated_tags = []
    for tag in tag_entry.split(','):
        if tag:
            parts = tag.split(':')
            if len(parts) == 3:  # Expected format [start:end:label]
                start, end, label = parts
                # Subtract 1 from start and 2 from end
                new_start = int(start) - 1
                new_end = int(end) - 2
                updated_tags.append(f"{new_start}:{new_end}:{label}")
    return ','.join(updated_tags)


def parse_tags(tags_string):
    """
    Parse the tags string into a list of tuples (start, end, entity_label).
    """
    if pd.isna(tags_string):
        return []
    return [(int(start), int(end), label) for start, end, label in 
            (tag.split(':') for tag in tags_string.split(',') if tag)]


def to_bio(tag_list):
    bio_tags = []
    for start, end, label in tag_list:
        bio_tags.append((start, end, f"B-{label}"))
        for i in range(start + 1, end):
            bio_tags.append((i, i+1, f"I-{label}"))
    return bio_tags

def tokenize_and_align_labels(texts, tokenizer, max_length=128):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    return encodings    

# Dataset class
class ClinicalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Model training function
def train_model(train_loader, model, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def plot_loss(epoch_losses, epoch):
    print("Plotting the Training Loss")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(epoch)), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

# Evaluation function
def evaluate_model(test_loader, model, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            # Convert predictions and labels to CPU and flatten
            predictions = predictions.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            # Flatten predictions and labels, filtering out ignored tokens
            active_accuracy = labels != -100
            all_predictions.extend(predictions[active_accuracy])
            all_labels.extend(labels[active_accuracy])
            
    return all_predictions, all_labels

def get_classification_report(test_loader, model):
    
    all_predictions, all_labels = evaluate_model(test_loader, model, device)
    
    # Define a simplified label mapping
    simplified_label_map = {
        0: 'O',
        1: 'treatment', 
        2: 'treatment', 
        3: 'chronic_disease', 
        4: 'chronic_disease', 
        5: 'cancer', 
        6: 'cancer', 
        7: 'allergy_name', 
        8: 'allergy_name'
    }

    # Create unique identifiers for new labels
    unique_label_ids = {v: k for k, v in enumerate(set(simplified_label_map.values()))}

    # Function to map original labels to their simplified form
    def map_labels_to_simplified(labels, label_map):
        return [unique_label_ids[label_map[label]] for label in labels if label in label_map]

    # Apply mapping to all_labels and all_predictions
    simplified_all_labels = map_labels_to_simplified(all_labels, simplified_label_map)
    simplified_all_predictions = map_labels_to_simplified(all_predictions, simplified_label_map)
    report = classification_report(simplified_all_labels,simplified_all_predictions,
        target_names=list(unique_label_ids.keys()),digits=3)
    # Compute and print the classification report for the simplified labels
    print(report)
    
    return report

def to_bio(tag_list, default_label='O'):
    """Converts tag lists to BIO format."""
    bio_tag_list = []
    last_end = 0
    for start, end, label in tag_list:
        # Fill in 'O' tags between entities
        if start > last_end:
            bio_tag_list.extend([(i, i + 1, default_label) for i in range(last_end, start)])
        bio_tag_list.append((start, end, f"B-{label}"))
        if end - start > 1:
            bio_tag_list.extend([(i, i + 1, f"I-{label}") for i in range(start + 1, end)])
        last_end = end
    return bio_tag_list

def create_labels(text, tags, tokenizer, label_map):
    """Creates token labels for NER based on BIO tagging."""
    tokenized_input = tokenizer(text, return_offsets_mapping=True, padding='max_length', max_length=128, truncation=True)
    label_ids = [-100] * len(tokenized_input['input_ids'])
    offset_mapping = tokenized_input['offset_mapping']

    for char_start, char_end, label in tags:
        token_start, token_end = None, None
        for i, (offset_start, offset_end) in enumerate(offset_mapping):
            if offset_start is None:  # Skip special tokens
                continue
            if char_start >= offset_start and char_start < offset_end:
                token_start = i
            if char_end > offset_start and char_end <= offset_end:
                token_end = i
        if token_start is not None and token_end is not None:
            label_ids[token_start] = label_map[label]
            for j in range(token_start + 1, token_end + 1):
                intra_label = label.replace('B-', 'I-')  # Change B- to I- for inside tags
                label_ids[j] = label_map.get(intra_label, label_map[label])  # Handle potential missing labels

    return label_ids

def process_for_ner(data, tokenizer, label_map):
    """Processes DataFrame for NER training."""
    data['bio_tags'] = data['parsed_tags'].apply(to_bio)
    data['token_labels'] = data.apply(lambda row: create_labels(row['text'], row['bio_tags'], tokenizer, label_map), axis=1)
    label= data['token_labels']

    return label

def adjust_ranges(tag_entry):
    if pd.isna(tag_entry):
        return tag_entry
    updated_tags = []
    for tag in tag_entry.split(','):
        if tag:
            parts = tag.split(':')
            if len(parts) == 3:
                start, end, label = parts
                new_start = int(start) - 1
                new_end = int(end) - 1
                updated_tags.append(f"{new_start}:{new_end}:{label}")
    return ','.join(updated_tags)

def extract_labels(tags):
    if pd.isna(tags):
        return []
    return re.findall(r'\d+:\d+:(\w+)', tags)

def process_dataset(dataset):
    dataset['new_tags'] = dataset['tags'].apply(adjust_ranges)
    dataset['labels'] = dataset['new_tags'].apply(extract_labels)
    dataset['labels'] = dataset['labels'].apply(lambda labels: sorted(set(labels)))
    dataset['label_combination'] = dataset['labels'].apply(lambda x: '_'.join(x))
    
    threshold = 10
    counts = dataset['label_combination'].value_counts()
    rare_labels = counts[counts <= threshold].index
    dataset['label_combination_adjusted'] = dataset['label_combination'].apply(
        lambda x: 'other' if x in rare_labels else x)
    
    return dataset

def create_train_test_datasets(dataset, test_size=0.2, random_state=42, save_path=None):
    train, test = train_test_split(
        dataset, 
        test_size=test_size, 
        stratify=dataset['label_combination_adjusted'], 
        random_state=random_state
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    if save_path:
        train.to_csv(f'{save_path}_train.csv', index=False)
        test.to_csv(f'{save_path}_test.csv', index=False)

    return train, test

def full_process_and_split_with_stratify(data_path, test_size=0.2, random_state=42, save_path=None):
    # Load the data
    dataset = pd.read_excel(data_path)
    # Process the data
    processed_dataset = process_dataset(dataset)
    # Create train-test datasets
    return create_train_test_datasets(processed_dataset, test_size, random_state, save_path)