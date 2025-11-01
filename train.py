#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from models import LeNetDigit

def load_data(csv_path, test_size=0.1, random_state=2):
    train = pd.read_csv(csv_path)
    Y = train["label"]
    X = train.drop("label", axis=1)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, stratify=Y
    )
    # reshape and convert
    X_train = X_train.values.reshape(-1, 28, 28, 1).astype(np.float32)
    X_val = X_val.values.reshape(-1, 28, 28, 1).astype(np.float32)
    X_train = torch.tensor(X_train).permute(0, 3, 1, 2)
    X_val = torch.tensor(X_val).permute(0, 3, 1, 2)
    Y_train = torch.tensor(Y_train.to_numpy(), dtype=torch.long)
    Y_val = torch.tensor(Y_val.to_numpy(), dtype=torch.long)
    return X_train, X_val, Y_train, Y_val

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print("Using device:", device)

    X_train, X_val, Y_train, Y_val = load_data(args.data, test_size=args.val_split)
    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = LeNetDigit().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {running_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    classes = [str(i) for i in range(10)]
    correct_pred = {c: 0 for c in classes}
    total_pred = {c: 0 for c in classes}

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            for label, pred in zip(labels, preds):
                if label == pred:
                    correct_pred[str(label.item())] += 1
                total_pred[str(label.item())] += 1

    overall_acc = 100.0 * correct / total
    print(f"Validation accuracy: {overall_acc:.2f}%")
    for classname, correct_count in correct_pred.items():
        acc = 100.0 * float(correct_count) / total_pred[classname] if total_pred[classname] > 0 else 0.0
        print(f"Class {classname}: {acc:.1f}%")

    # save model
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, "digit_model.pth")
    torch.save(model.state_dict(), save_path)
    print("Saved model to", save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="train.csv", help="path to train CSV (unzipped)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--out-dir", type=str, default="models")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()
    train(args)
