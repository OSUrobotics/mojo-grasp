import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(suppress=True, precision=5, floatmode='fixed')

def init_wandb():
    wandb.init(
        project="autoencoder-project",
        config={
            "input_dim": 72,     
            "latent_dim": 16,
            "output_dim": 55,   
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 4096,
            "plane_loss_weight": 0.000
        }
    )
    return wandb.config

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, dropout_prob=0.0):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_prob),
            nn.Linear(128, latent_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_prob),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

    def decode(self, latent):
        return self.decoder(latent)

class WeightedMSELoss(nn.Module):
    def __init__(self, weight_tensor):
        super(WeightedMSELoss, self).__init__()
        self.weight_tensor = weight_tensor

    def forward(self, pred, target):
        loss = (self.weight_tensor * (pred - target) ** 2).mean()
        return loss

def distance_from_plane_loss(recon, target):
    target = target.view(-1, 24, 3)
    recon = recon.view(-1, 24, 3)

    centroid = target.mean(dim=1, keepdim=True)
    centered = target - centroid
    _, _, v = torch.linalg.svd(centered)
    normal = v[:, -1, :].unsqueeze(1)  # shape: (B, 1, 3)

    distances = torch.abs(torch.sum((recon - centroid) * normal, dim=2))
    return distances.mean()

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            _, reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_y)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def train_model(model, train_loader, test_loader, criterion, optimizer, config):
    best_loss = float("inf")
    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            _, reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_y)
            #loss += config.plane_loss_weight * distance_from_plane_loss(reconstructed, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        avg_test_loss = evaluate_model(model, test_loader, criterion)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "test_loss": avg_test_loss,
        })

        print(
            f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | "
            f"Test Loss: {avg_test_loss:.4f} | "
        )

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), "/home/ubuntu/Mojograsp/mojo-grasp/demos/rl_demo/test_best_autoencoder_16.pth")
            print(f"Best model saved with test loss: {best_loss:.8f}")

def test_single_row(model, df, input_scaler, output_scaler, input_dim, output_dim, row_index=None):
    if row_index is None:
        sample_df = df.sample(1)
        actual_index = sample_df.index[0]
    else:
        sample_df = df.iloc[[row_index]]
        actual_index = row_index

    # Get input (dynamic = last 72 columns)
    row_input = sample_df.iloc[:, -input_dim:].values

    # Get ground truth quaternion (columns 3 to 6 inclusive)
    ground_truth = sample_df.iloc[:, 3:7].values

    # Scale input and convert to tensor
    scaled_row_input = input_scaler.transform(row_input)
    input_tensor = torch.tensor(scaled_row_input, dtype=torch.float32)

    # Forward pass
    with torch.no_grad():
        _, reconstruction = model(input_tensor)

    # Unscale output (predicted quaternion)
    reconstruction_np = reconstruction.cpu().numpy()
    reconstruction_unscaled = output_scaler.inverse_transform(reconstruction_np)

    ground_truth_quat = ground_truth[0]
    reconstructed_quat = reconstruction_unscaled[0]

    # Normalize quaternions (to compare orientation properly)
    gt_norm = ground_truth_quat / np.linalg.norm(ground_truth_quat)
    pred_norm = reconstructed_quat / np.linalg.norm(reconstructed_quat)

    # Dot product similarity (-1 to 1)
    quat_dot = np.dot(gt_norm, pred_norm)

    # Angular distance (in degrees)
    quat_dot = np.clip(quat_dot, -1.0, 1.0)  # ensure within acos domain
    angular_distance_deg = 2 * np.arccos(abs(quat_dot)) * 180 / np.pi

    print(f"\n[Row {actual_index}]")
    print("Ground Truth Quaternion:     ", np.round(gt_norm, 5))
    print("Reconstructed Quaternion:    ", np.round(pred_norm, 5))
    print(f"Dot Product Similarity:      {quat_dot:.5f}")
    print(f"Angular Distance (degrees):  {angular_distance_deg:.3f}°\n")

def save_model(model, filename="untrained_autoencoder.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")

def load_trained_model(model_path, input_dim, latent_dim, output_dim):
    model = Autoencoder(input_dim, latent_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_csv(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, input_dim, output_dim, test_size=0.25):
    # Dynamic input (last 72 columns)
    data_x = df.iloc[:, -input_dim:].values

    # Static output (first 55 columns)
    data_y = df.iloc[:, :output_dim].values

    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()

    scaled_x = input_scaler.fit_transform(data_x)
    scaled_y = output_scaler.fit_transform(data_y)

    X_train, X_test, y_train, y_test = train_test_split(scaled_x, scaled_y,
                                                        test_size=test_size, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.float32))

    return train_dataset, test_dataset, input_scaler, output_scaler

def save_scalers(input_scaler, output_scaler, input_filename="/home/ubuntu/Mojograsp/mojo-grasp/demos/rl_demo/test_input_scaler.pkl", output_filename="/home/ubuntu/Mojograsp/mojo-grasp/demos/rl_demo/test_output_scaler.pkl"):
    with open(input_filename, "wb") as f:
        pickle.dump(input_scaler, f)
    with open(output_filename, "wb") as f:
        pickle.dump(output_scaler, f)
    print(f"Input scaler saved as {input_filename} and output scaler saved as {output_filename}")

def load_scalers(input_filename="/home/ubuntu/Mojograsp/mojo-grasp/demos/rl_demo/test_input_scaler.pkl", output_filename="/home/ubuntu/Mojograsp/mojo-grasp/demos/rl_demo/test_output_scaler.pkl"):
    with open(input_filename, "rb") as f:
        input_scaler = pickle.load(f)
    with open(output_filename, "rb") as f:
        output_scaler = pickle.load(f)
    print(f"Input scaler loaded from {input_filename} and output scaler loaded from {output_filename}")
    return input_scaler, output_scaler

def main():

    # Use this block only if you want to load and test an already trained model:

    df = load_csv("test_final_data.csv")
    loaded_input_scaler, loaded_output_scaler = load_scalers("/home/ubuntu/Mojograsp/mojo-grasp/demos/rl_demo/test_input_scaler.pkl", "/home/ubuntu/Mojograsp/mojo-grasp/demos/rl_demo/test_output_scaler.pkl")
    loaded_model = load_trained_model("/home/ubuntu/Mojograsp/mojo-grasp/demos/rl_demo/test_best_autoencoder_16.pth", 72, 16, 55)
    
    # Test a single (random) row from the original dataframe multiple times
    for i in range(5):
        test_single_row(loaded_model, df, loaded_input_scaler, loaded_output_scaler,72, 55)
    """


    config = init_wandb()
    df = load_csv("/home/ubuntu/Mojograsp/mojo-grasp/test_final_data.csv")
    
    train_dataset, test_dataset, input_scaler, output_scaler = preprocess_data(
        df, config.input_dim, config.output_dim
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = Autoencoder(config.input_dim, config.latent_dim, config.output_dim)

    save_scalers(input_scaler, output_scaler)

    weights = torch.ones(config.output_dim, dtype=torch.float32)
    weights[3:7] = 10.0  

    criterion = WeightedMSELoss(weights)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    wandb.watch(model, log="all")
    train_model(model, train_loader, test_loader, criterion, optimizer, config)
    wandb.finish()
    print("Training complete with Weighted MSELoss!")
    """

if __name__ == "__main__":
    main()




