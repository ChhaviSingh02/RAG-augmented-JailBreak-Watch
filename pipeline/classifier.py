"""
Classifier Training — Fine-tune DistilBERT for jailbreak detection
This gives you measurable F1/precision/recall for the research paper
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import json
from pathlib import Path
import numpy as np

# Configuration
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
SAVE_PATH = "models/classifier"

class JailbreakDataset(Dataset):
    """PyTorch Dataset for jailbreak prompts"""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data():
    """Load train and test splits"""
    print("\n[1/5] Loading data...")
    
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    print(f"  Train jailbreak ratio: {(train_df['label']==1).sum()/len(train_df):.2%}")
    print(f"  Test jailbreak ratio: {(test_df['label']==1).sum()/len(test_df):.2%}")
    
    return train_df, test_df

def create_dataloaders(train_df, test_df, tokenizer):
    """Create PyTorch DataLoaders"""
    print("\n[2/5] Creating DataLoaders...")
    
    train_dataset = JailbreakDataset(
        train_df['text'].values,
        train_df['label'].values,
        tokenizer,
        MAX_LENGTH
    )
    
    test_dataset = JailbreakDataset(
        test_df['text'].values,
        test_df['label'].values,
        tokenizer,
        MAX_LENGTH
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"  ✓ Train batches: {len(train_loader)}")
    print(f"  ✓ Test batches: {len(test_loader)}")
    
    return train_loader, test_loader

def train_model(model, train_loader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="  Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def evaluate_model(model, test_loader, device):
    """Evaluate on test set"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return np.array(predictions), np.array(true_labels)

def main():
    print("=" * 60)
    print("JAILBREAK DETECTION — CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    train_df, test_df = load_data()
    
    # Load tokenizer and model
    print(f"\n[2/5] Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    ).to(device)
    print("  ✓ Model loaded")
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(train_df, test_df, tokenizer)
    
    # Setup optimizer and scheduler
    print("\n[3/5] Setting up training...")
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    print(f"  Total training steps: {total_steps}")
    
    # Training loop
    print(f"\n[4/5] Training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        train_loss, train_acc = train_model(
            model, train_loader, optimizer, scheduler, device
        )
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Accuracy: {train_acc:.4f}")
    
    # Evaluation
    print("\n[5/5] Final evaluation on test set...")
    predictions, true_labels = evaluate_model(model, test_loader, device)
    
    # Classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(
        true_labels, 
        predictions, 
        target_names=['Safe', 'Jailbreak'],
        digits=4
    )
    print(report)
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predictions)
    print(f"              Predicted")
    print(f"              Safe  Jailbreak")
    print(f"Actual Safe     {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"       Jailbreak {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # Save model
    print(f"\nSaving model to {SAVE_PATH}...")
    Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("  ✓ Model saved")
    
    # Save metrics for paper
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary'
    )
    
    metrics = {
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "test_accuracy": float((predictions == true_labels).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist()
    }
    
    with open(f"{SAVE_PATH}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  ✓ Metrics saved to {SAVE_PATH}/metrics.json")
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nKey Metrics (for your paper):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"\nModel ready for inference!")

if __name__ == "__main__":
    main()