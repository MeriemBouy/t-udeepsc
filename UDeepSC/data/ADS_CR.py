import os
import torch
import random
from torch.utils.data import Dataset
from transformers import BertTokenizer

class ADS_CR(Dataset):
    def __init__(self, root=False, train=True, binary=True, if_class=True, max_length=128):
        """
        Custom Dataset mimicking the SST_CR parameters for UDeepSC integration.
        """
        # 'root' parameter
        base_dir = './data/' if root is False else root
        file_path = os.path.join(base_dir, 'autonomous_driving_semantic_dataset.txt')
        
        # 'if_class' parameter
        self.if_class = if_class
        if not self.if_class:
            raise NotImplementedError("This loader is currently only designed for classification tasks.")
            
        # 'binary' parameter
        # We accept 'binary' to match the signature, but we inherently use the 
        # multi-class labels of the ADS dataset
        
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        self.texts = []
        self.labels = []
        
        # Parse the custom dataset
        self._load_and_parse_file(file_path)

        # Shuffle the data deterministically
        combined = list(zip(self.texts, self.encoded_labels))
        random.seed(42) # Locks the random shuffle so Train and Test splits don't overlap
        random.shuffle(combined)
        self.texts, self.encoded_labels = zip(*combined)
        
        # Convert tuples back to lists
        self.texts = list(self.texts)
        self.encoded_labels = list(self.encoded_labels)
        
        # 'train' parameter (Train/Val Split)
        # We perform a deterministic 80/20 split based on the 'train' boolean
        split_idx = int(len(self.texts) * 0.8)
        if train:
            self.texts = self.texts[:split_idx]
            self.encoded_labels = self.encoded_labels[:split_idx]
        else:
            self.texts = self.texts[split_idx:]
            self.encoded_labels = self.encoded_labels[split_idx:]

    def _load_and_parse_file(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        blocks = content.strip().split('\n\n')
        unique_labels = set()
        
        for block in blocks:
            if not block.strip():
                continue
                
            if 'Description:' in block and 'SemanticClass:' in block:
                parts = block.split('SemanticClass:')
                desc_part = parts[0].replace('Description:', '').strip()
                label_part = parts[1].strip()
                
                self.texts.append(desc_part)
                self.labels.append(label_part)
                unique_labels.add(label_part)
                
        # Create alphabetical label mapping
        self.label_map = {label: idx for idx, label in enumerate(sorted(list(unique_labels)))}
        self.encoded_labels = [self.label_map[l] for l in self.labels]
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.encoded_labels[idx]
        
        # Format tokens for BERT encoder inside UDeepSC
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return encoding['input_ids'].flatten(), torch.tensor(label, dtype=torch.long)