import argparse
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
import numpy as np
import pandas as pd
import json
import pdb

def tokenize(examples, tokenizer, text_key='text'):
    return tokenizer(examples[text_key], truncation=True, padding="max_length", max_length=512)

def create_label(examples):
    examples['label'] = 1 if examples['Sentiment'] == "Positive" else 0
    return examples


def load_data(tokenizer, data, csv_path, mode):
    if data == "imdb":
        dataset = load_dataset("imdb")
        dataset = dataset.map(lambda x: tokenize(x, tokenizer=tokenizer), batched=True, num_proc=16)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        return dataset[mode]
    elif data == 'imdb_contrast':
        df = pd.read_csv(csv_path)
        df["label"] = np.where(df['label'] == "Negative", 0, 1)
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(lambda x: tokenize(x, tokenizer=tokenizer), batched=True, num_proc=16)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        return dataset
    elif data == "imdb_adv":
        dataset = load_dataset("tasksource/counterfactually-augmented-imdb")
        dataset = dataset.map(lambda x: tokenize(x, tokenizer=tokenizer, text_key="Text"), batched=True, num_proc=16)
        dataset = dataset.map(create_label, num_proc=16)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        return dataset[mode]
    else:
        raise ValueError(f"unrecognized dataset {data}")

def train_model(dataset, model, device, save_dir, epochs=2):
    print(f"Training samle size: {len(dataset)}")
    # Data loader
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=16)
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)
    step = 0
    # Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            step += 1
            if step % 100 == 0:
                print(f"step: {step}, training loss: {loss}")
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} - Average Training loss: {avg_train_loss:.4f}")

        # Save the model after each epoch
        if save_dir:
            model_save_path = os.path.join(save_dir, f"bert_sentiment_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

def token_ids_to_text(tokenizer, input_ids):
    text = tokenizer.decode(input_ids)
    # first remove [CLS]
    text = text.replace("[CLS] ", "")
    # next get text till [SEP] token
    sentence_end_idx = text.index("[SEP]")
    return text[:sentence_end_idx]


def eval_model(dataset, model, device, tokenizer, save_wrong_predction_dir):
    test_dataloader = DataLoader(dataset, batch_size=8, num_workers=16)
    model.eval()
    all_labels = []
    all_preds = []
    wrong_examples = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = torch.Tensor(batch["label"]).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            if save_wrong_predction_dir:
                ## Gather indices of incorrect predictions
                wrong_indices = (predictions != labels).nonzero(as_tuple=True)[0]
                # Store wrong examples
                for idx in wrong_indices:
                    wrong_examples.append({
                        "input_text": token_ids_to_text(tokenizer, input_ids[idx].cpu().numpy().tolist()),
                        "real_label": labels[idx].item(),
                        "predicted_label": predictions[idx].item()
                    })
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
    if save_wrong_predction_dir:
        # Save wrong examples to a file
        with open(os.path.join(save_wrong_predction_dir, "wrong_examples.json"), "w") as f:
            json.dump(wrong_examples, f, indent=4)
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

def run(args):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    if args.load_weights:
        model.load_state_dict(torch.load(args.load_weights))
        print(f"Loaded weights from {args.load_weights}")
    model.to(device)
    # Do training
    if args.do_train:
        dataset = load_data(tokenizer, args.data, args.csv_path, mode="train")
        train_model(dataset, model, device, save_dir=args.save_dir)

    elif args.do_eval:
        dataset = load_data(tokenizer, args.data, args.csv_path, mode="test")
        eval_model(dataset, model, device, tokenizer=tokenizer, save_wrong_predction_dir=args.save_wrong_predction_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="the data to be used, can be either 'imdb', 'imdb_contrast', 'imdb_adv'")
    parser.add_argument("--csv_path", type=str, help="the path of the data csv file if it is available")
    parser.add_argument("--do_train", action="store_true", help="whether to do training")
    parser.add_argument("--do_eval", action="store_true", help="whether to do evaluation")
    parser.add_argument("--load_weights",default=None, type=str, help="the weights of the file to be loaded if any")
    parser.add_argument("--save_dir", default=None, help="the directory to save model's weights")
    parser.add_argument("--save_wrong_predction_dir", default=None, help="the directory to save model's wrong prediction samples during evaluation")
    args = parser.parse_args()
    run(args)