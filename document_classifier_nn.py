import sys
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 8)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.out(x)


def load_data():
    url = "https://s3.amazonaws.com/hr-testcases/597/assets/trainingdata.txt"
    labels = []
    documents = []
    i = 0

    response = requests.get(url, stream=True)
    response.raise_for_status()

    for line in response.iter_lines(decode_unicode=True):
        if i == 0:
            i = i + 1
            continue

        label, document = line.strip().split(" ", 1)
        labels.append(int(label))
        documents.append(document)

    vectorizer = CountVectorizer(stop_words="english", max_features=1000)
    X = vectorizer.fit_transform(documents).toarray()
    return X, torch.tensor(labels) - 1, vectorizer


def train_model():
    X, y, vectorizer = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val = torch.tensor(X_train, dtype=torch.float32), torch.tensor(
        X_val, dtype=torch.float32
    )
    Y_train, Y_val = torch.tensor(y_train, dtype=torch.long), torch.tensor(
        y_val, dtype=torch.long
    )

    input_size = X_train.shape[1]

    model = Model(input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    epochs = 1000
    losses = []
    for i in range(epochs):
        # forward propagate
        y_pred = model.forward(X_train)

        # measure the loss
        loss = criterion(y_pred, y_train)
        losses.append(loss.detach().numpy())  # converts the torch back to numpy

        if i % 10 == 0:
            print(f"Epoch: {i} - Loss: {loss}")

        # backwards propagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        _, predictions = torch.max(val_outputs, 1)
        accuracy = (predictions == y_val).float().mean()

        print(accuracy)

    return model


if __name__ == "__main__":
    model = train_model()

    for line in sys.stdin:
        print(f"Line: {line.strip()}")
