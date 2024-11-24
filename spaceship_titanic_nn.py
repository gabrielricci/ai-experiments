import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(14, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        return self.out(x)


def load_data(filepath):
    data = pd.read_csv(filepath)

    data[["Deck", "CabinNum", "Side"]] = data["Cabin"].str.split("/", expand=True)

    label_encoder = LabelEncoder()
    data["Deck"] = label_encoder.fit_transform(data["Deck"].fillna("Unknown"))
    data["Side"] = label_encoder.fit_transform(data["Side"].fillna("Unknown"))
    data["Destination"] = label_encoder.fit_transform(
        data["Destination"].fillna("Unknown")
    )
    data["HomePlanet"] = label_encoder.fit_transform(
        data["HomePlanet"].fillna("Unknown")
    )

    data = data.fillna(value=0)

    data["Bills"] = data["FoodCourt"] + data["RoomService"] + data["ShoppingMall"]

    data["VIP"] = data["VIP"].astype(float)
    data["CryoSleep"] = data["CryoSleep"].astype(float)
    data["Destination"] = data["Destination"].astype(float)

    return data.drop(
        ["PassengerId", "Name", "Cabin"],
        axis=1,
    )


def load_train_data(filepath):
    data = load_data(filepath)

    y = data["Transported"].astype(float).values
    X = data.drop(["Transported"], axis=1).astype(float).values

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    return X, y


def load_test_data(filepath):
    data = load_data(filepath)

    X = data.astype(float).values

    return torch.tensor(X, dtype=torch.float32)


def train_model(X_train, y_train):
    torch.manual_seed(41)

    model = Model()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10000
    losses = []

    model.train()

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

    return model, criterion


def test_model(model, criterion, X_val, y_val):
    # Evaluation on the validation set after training
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val).item()
        print(f"Validation Loss: {val_loss}")

        # Convert logits to probabilities using sigmoid and calculate accuracy
        val_pred = torch.sigmoid(val_pred)
        predicted_labels = (val_pred > 0.5).float()  # Convert probabilities to 0 or 1

        print(X_val)
        print(predicted_labels)

        # Calculate accuracy
        accuracy = (predicted_labels == y_val).float().mean()
        print(f"Validation Accuracy: {accuracy.item() * 100:.2f}%")


def predict(model, X_val):
    # Evaluation on the validation set after training
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)

        # Convert logits to probabilities using sigmoid and calculate accuracy
        val_pred = torch.sigmoid(val_pred)
        predicted_labels = (val_pred > 0.5).float()

        return predicted_labels


if __name__ == "__main__":
    X_train, y_train = load_train_data("spaceship_titanic_train.csv")
    X_test = load_test_data("spaceship_titanic_test.csv")

    model, criterion = train_model(X_train, y_train)
    predictions = predict(model, X_test)

    with open("spaceship_titanic_test.csv", "r") as file:
        i = 0
        for line in file:
            if i == 0:
                print("PassengerId,Transported")
                i = i + 1
                continue

            passengerId, rest = line.strip().split(",", 1)
            transported = predictions[i - 1][0] > 0
            print(f"{passengerId},{transported}")
            i = i + 1
