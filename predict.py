import torch
import numpy as np
from main import Model, device, mean, std, y_mean, y_std
from main import continuous_columns, onehot_columns, transform

input_dim = len(continuous_columns) + len(onehot_columns)
model = Model(input_dim, 1)
model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

def predict(input_dict):
    import pandas as pd
    row = pd.DataFrame([input_dict])
    
    vocab = ["NEAR BAY", "INLAND", "NEAR OCEAN", "ISLAND", "<1H OCEAN"]
    for category in vocab:
        clean_name = transform(category)
        row["ocean_" + clean_name] = (row["ocean_proximity"] == category).astype(int)
    row = row.drop("ocean_proximity", axis=1)
    
    for col in continuous_columns:
        row[col] = (row[col] - mean[col]) / std[col]
    row = row[continuous_columns + onehot_columns]

    x = torch.tensor(row.to_numpy(), dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(x).cpu().numpy()
    return float(pred[0][0]) * y_std + y_mean

result = predict({
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41.0,
    "total_rooms": 880.0,
    "total_bedrooms": 129.0,
    "population": 322.0,
    "households": 126.0,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
})

print(f"Predicted house value: ${result:,.0f}")
