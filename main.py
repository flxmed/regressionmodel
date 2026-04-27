import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn

# Hyperparameters
HIDDEN_SIZE = 128
DROPOUT = 0.2
EPOCHS = 5000
LR = 1e-3
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42

df = pd.read_csv("housing.csv")

y = df["median_house_value"]
x = df.drop("median_house_value", axis=1)

mask = x["total_bedrooms"].notnull()

x = x[mask]
y = y[mask]

x = x.copy()
y = y.copy()

x = x.reset_index(drop=True)
y = y.reset_index(drop=True)

vocab = x["ocean_proximity"].unique()

def transform(category):
    return category.replace(" ", "_").replace("<", "").replace(">", "").lower()

for category in vocab:
    clean_name = transform(category)
    column_name = "ocean_" + clean_name
    x[column_name] = (x["ocean_proximity"] == category).astype(int)
x = x.drop("ocean_proximity", axis=1)

def train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True):
    n = len(x)
    split = int(n * (1 - test_size))
    indices = list(range(n))
    if shuffle:
        random.seed(random_state)
        random.shuffle(indices)
    train_indices = indices[:split]
    test_indices = indices[split:]
    x_train = x.iloc[train_indices]
    y_train = y.iloc[train_indices]
    x_test = x.iloc[test_indices]
    y_test = y.iloc[test_indices]
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = train_test_split(x, y)

onehot_columns = [col for col in x_train.columns if col.startswith("ocean_")]
continuous_columns = [col for col in x_train.columns if not col.startswith("ocean_")]

mean = x_train[continuous_columns].mean(axis=0)
std = x_train[continuous_columns].std(axis=0)

x_train_normalized = x_train[continuous_columns].copy()
x_test_normalized = x_test[continuous_columns].copy()

y_mean = y_train.mean()
y_std = float(y_train.std())

y_train_normalized = (y_train - y_mean) / y_std
y_test_normalized = (y_test - y_mean) / y_std

x_train_normalized[onehot_columns] = x_train[onehot_columns]
x_test_normalized[onehot_columns] = x_test[onehot_columns]

for col in continuous_columns:
    x_train_normalized[col] = (x_train_normalized[col] - mean[col]) / std[col]
    x_test_normalized[col] = (x_test_normalized[col] - mean[col]) / std[col]

x_train_np = x_train_normalized.to_numpy()
y_train_np = y_train_normalized.to_numpy().reshape(-1, 1)
x_test_np = x_test_normalized.to_numpy()
y_test_np = y_test_normalized.to_numpy().reshape(-1, 1)

x_train_tensor = torch.tensor(x_train_np, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test_np, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)


input_dim = x_train_tensor.shape[1]
output_dim = 1

class Model(nn.Module):
    def __init__(self, in_features, out_features, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, out_features)
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = torch.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)
        return x
    
model = Model(input_dim, output_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
x_train_tensor = x_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
x_test_tensor = x_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

def train(model, x_train, y_train, x_test, y_test, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    stop = False
    patience_counter = 0
    for epoch in range(epochs):
        if stop:
            break
        model.train()
        indices = torch.randperm(x_train.shape[0], device=x_train.device)
        num_batches = (x_train.shape[0] + batch_size - 1) // batch_size
        epoch_loss = 0
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            
            x_batch = x_train[indices[start:end]]
            y_batch = y_train[indices[start:end]]

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss / num_batches}")
            model.eval()
            with torch.no_grad():
                y_test_pred = model(x_test)
                test_loss = criterion(y_test_pred, y_test)
                scheduler.step(test_loss)
                if test_loss.item() < best_loss:
                    best_loss = test_loss.item()
                    torch.save(model.state_dict(), "best_model.pth")
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= 20:
                    print("Early stopping triggered.")
                    stop = True
                    break
                
                test_rmse = torch.sqrt(criterion(y_test_pred, y_test)) * y_std
                print(f"Epoch {epoch}, Test Loss: {test_loss.item()}, Test RMSE: {test_rmse.item()}")

def predict(model, x):
        model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
            y_pred = model(x_tensor)
            y_pred = y_pred.cpu().numpy() * y_std + y_mean
            return y_pred
            
if __name__ == '__main__':
    
    train(model, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor)
    
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test_tensor)
        test_loss = nn.MSELoss()(y_test_pred, y_test_tensor)
        test_rmse = torch.sqrt(test_loss) * y_std
        print(f"Final Test Loss: {test_loss.item()}, Final Test RMSE: {test_rmse.item()}")

    
        
    sample_input = x_test_normalized.iloc[0].to_numpy().reshape(1, -1)
    predicted_value = predict(model, sample_input)
    print(f"Predicted median house value: {predicted_value[0][0]}, Actual median house value: {y_test.iloc[0]}")










