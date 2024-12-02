import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

class ReviewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    losses = []
    correct_predictions = 0

    progress_bar = tqdm(data_loader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss.mean() 
        logits = outputs.logits

        losses.append(loss.item())
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.set_postfix({"Batch Loss": loss.item()})

    progress_bar.close()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)


def evaluate_epoch(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss.mean() 
            logits = outputs.logits

            losses.append(loss.item())
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=[str(i + 1) for i in range(5)])
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses), accuracy, report

def main(args):
    print("Loading dataset...")
    df = pd.read_json(args.input_file, lines=True)  
    df = df[['review_text', 'rating']]
    print("Dataset loaded!")
    df['rating'] = df['rating'] - 1

    train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['review_text'].tolist(), df['rating'].tolist(), test_size=0.2, random_state=42
    )


    MODEL_NAME = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    MAX_LENGTH = 128

    train_dataset = ReviewsDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = ReviewsDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print("loading model...")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)
    print("Model loaded!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)


    # Training loop remains the same
    print("Training model...")
    EPOCHS = 1
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Train loss: {train_loss}, Train accuracy: {train_acc}")

        val_acc, val_loss, val_accuracy, val_report = evaluate_epoch(model, val_loader, device)
        print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")
        print("Classification Report:")
        print(val_report)
    # Save the model
    model.module.save_pretrained(args.output_dir)  # Use model.module for DataParallel
    tokenizer.save_pretrained(args.output_dir)

    print("Model training complete and saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--train_size", type=float, default=0.9)
    args = parser.parse_args()

    main(args)