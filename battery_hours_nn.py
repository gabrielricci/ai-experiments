import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


class Model(nn.Module):
    # Input Layer (1) -> Hidden Layer (2) -> Output Layer (1)
    def __init__(self):
        super(Model, self).__init__()
        self.out = nn.Linear(1, 1)

    def forward(self, x):
        # x = torch.relu(self.fc1(x))
        x = self.out(x)
        return torch.minimum(x, torch.tensor(8.0))


def train_model():
    torch.manual_seed(41)
    model = Model()

    url = "https://s3.amazonaws.com/hr-testcases/399/assets/trainingdata.txt"
    data = pd.read_csv(url, header=None)
    x = data.drop(1, axis=1).values
    y = data[1].values

    # convert to Tensors
    x_train = torch.tensor(x, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 100000
    losses = []
    for i in range(epochs):
        # forward propagate
        y_pred = model.forward(x_train)

        # measure the loss
        loss = criterion(y_pred, y_train)
        losses.append(loss.detach().numpy())  # converts the torch back to numpy

        if i % 10 == 0:
            print(f"Epoch: {i} - Loss: {loss}")

        # backwards propagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


if __name__ == "__main__":
    model = train_model()

    # Test the model
    with torch.no_grad():
        test_input = torch.tensor([[3.0], [6.0], [10.0]], dtype=torch.float32)
        test_output = model(test_input)

        print("Test Inputs:", test_input.squeeze().tolist())
        print("Predicted Outputs:", test_output.squeeze().tolist())
