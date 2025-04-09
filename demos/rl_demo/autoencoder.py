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
import torch.nn.functional as F

np.set_printoptions(suppress=True, precision=6, floatmode='fixed')

def init_wandb():
    wandb.init(
        project="autoencoder_last-project",
        config={
            "input_dim": 72,     
            "latent_dim": 16,
            "output_dim": 72,   
            "learning_rate": 0.001,
            "epochs": 200,
            "batch_size": 1024,
        }
    )
    return wandb.config

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(Autoencoder, self).__init__()

        # Encoder layers
        self.enc1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)

        self.enc2 = nn.Linear(32, 32)
        self.bn2 = nn.BatchNorm1d(32)

        self.enc3 = nn.Linear(32, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.enc4 = nn.Linear(32, latent_dim)  

        # Decoder layers
        self.dec1 = nn.Linear(latent_dim, 32)
        self.dbn1 = nn.BatchNorm1d(32)

        self.dec2 = nn.Linear(32, 32)
        self.dbn2 = nn.BatchNorm1d(32)

        self.dec3 = nn.Linear(32, 32)
        self.dbn3 = nn.BatchNorm1d(32)

        self.dec4 = nn.Linear(32, output_dim)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.bn1(self.enc1(x)))  
        x2 = F.relu(self.bn2(self.enc2(x1)))
        x3 = F.relu(self.bn3(self.enc3(x2)))
        latent = torch.tanh(self.enc4(x3))   

        # Decoder with skip connections
        d1 = F.relu(self.dbn1(self.dec1(latent)))          
        d2 = F.relu(self.dbn2(self.dec2(d1 + x3)))       
        d3 = F.relu(self.dbn3(self.dec3(d2 + x2)))         
        reconstructed = torch.tanh(self.dec4(d3 + x1))     

        return latent, reconstructed

    def decode(self, latent):
        d1 = F.relu(self.dbn1(self.dec1(latent)))
        d2 = F.relu(self.dbn2(self.dec2(d1)))  # No skip in decode-only
        d3 = F.relu(self.dbn3(self.dec3(d2)))
        reconstructed = torch.tanh(self.dec4(d3))
        return reconstructed

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
            f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.5f} | "
            f"Test Loss: {avg_test_loss:.5f} | "
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

    # Scale input and convert to tensor
    scaled_row_input = input_scaler.transform(row_input)
    input_tensor = torch.tensor(scaled_row_input, dtype=torch.float32)

    # Forward pass
    with torch.no_grad():
        _, reconstruction = model(input_tensor)

    # Unscale output (reconstructed XYZ)
    reconstruction_np = reconstruction.cpu().numpy()
    reconstruction_unscaled = output_scaler.inverse_transform(reconstruction_np)

    # Reshape both ground truth and reconstruction to (24, 3)
    gt_points = row_input.reshape(24, 3)
    pred_points = reconstruction_unscaled.reshape(24, 3)

    # Plot both sets of points on the same 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], label='Ground Truth', marker='o')
    ax.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], label='Reconstruction', marker='^')


    ax.set_zlim([0.025, 0.075])

    ax.set_title(f"3D Shape Reconstruction - Row {actual_index}")
    ax.legend()
    plt.show()


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
    data_y = df.iloc[:, -input_dim:].values

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
    loaded_model = load_trained_model("/home/ubuntu/Mojograsp/mojo-grasp/demos/rl_demo/test_best_autoencoder_16.pth", 72, 16, 72)
    
    # Test a single (random) row from the original dataframe multiple times
    for i in range(15):
        test_single_row(loaded_model, df, loaded_input_scaler, loaded_output_scaler,72, 72)
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

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    wandb.watch(model, log="all")
    train_model(model, train_loader, test_loader, criterion, optimizer, config)
    wandb.finish()
    print("Training complete with Weighted MSELoss!")
    """

if __name__ == "__main__":
    main()




