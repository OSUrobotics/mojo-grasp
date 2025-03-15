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

def init_wandb():
    wandb.init(
        project="autoencoder-project",
        config={
            "input_dim": 72,
            "latent_dim": 16,
            "output_dim": 55,
            "learning_rate": 0.001,
            "epochs": 50,
            "batch_size": 64,
        }
    )
    return wandb.config

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, latent_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


### Weighted MSE Loss ###
class WeightedMSELoss(nn.Module):
    """
    Custom Weighted MSE:
    - Each output dimension is scaled by a weight in `weight_tensor`.
    - If `weight_tensor` is size (output_dim,), each dimension i is weighted by weight_tensor[i].
    """
    def __init__(self, weight_tensor):
        super(WeightedMSELoss, self).__init__()
        self.weight_tensor = weight_tensor

    def forward(self, pred, target):
        # (pred - target)^2 has shape [batch_size, output_dim]
        # Broadcasting: weight_tensor of shape [output_dim] will
        # automatically expand to match [batch_size, output_dim].
        loss = (self.weight_tensor * (pred - target) ** 2).mean()
        return loss


def save_model(model, filename="untrained_autoencoder.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")

def load_csv(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, input_dim, output_dim, test_size=0.25):
    # Extract output (first `output_dim` columns) and input (next `input_dim` columns)
    data_y = df.iloc[:, :output_dim].values
    data_x = df.iloc[:, output_dim:output_dim + input_dim].values

    # Create separate scalers for inputs and outputs
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()

    # Fit and transform each separately
    scaled_x = input_scaler.fit_transform(data_x)
    scaled_y = output_scaler.fit_transform(data_y)

    # Split the scaled data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_x, scaled_y, 
                                                        test_size=test_size, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.float32))
    
    return train_dataset, test_dataset, input_scaler, output_scaler

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
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_test_loss = evaluate_model(model, test_loader, criterion)
        
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "test_loss": avg_test_loss})
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
        
        # Save the best model
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), "test_best_autoencoder_16.pth")
            print(f"Best model saved with test loss: {best_loss:.5f}")

def load_trained_model(model_path, input_dim, latent_dim, output_dim):
    model = Autoencoder(input_dim, latent_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  
    return model

def save_scalers(input_scaler, output_scaler, input_filename="input_scaler.pkl", output_filename="output_scaler.pkl"):
    with open(input_filename, "wb") as f:
        pickle.dump(input_scaler, f)
    with open(output_filename, "wb") as f:
        pickle.dump(output_scaler, f)
    print(f"Input scaler saved as {input_filename} and output scaler saved as {output_filename}")

def load_scalers(input_filename="input_scaler.pkl", output_filename="output_scaler.pkl"):
    with open(input_filename, "rb") as f:
        input_scaler = pickle.load(f)
    with open(output_filename, "rb") as f:
        output_scaler = pickle.load(f)
    print(f"Input scaler loaded from {input_filename} and output scaler loaded from {output_filename}")
    return input_scaler, output_scaler

def test_single_row(model, df, input_scaler, output_scaler, input_dim, output_dim, row_index=None):
    """
    Tests a single row from the dataframe:
    - If row_index is None, a random row is chosen.
    - The row is scaled using the provided input_scaler.
    - The appropriate input slice is passed through the model.
    - Both latent and reconstructed outputs are printed.
    """
# 1) Pick the row
    if row_index is None:
        sample_df = df.sample(1)  # Random row
        actual_index = sample_df.index[0]  # Store the actual index used
    else:
        sample_df = df.iloc[[row_index]]  # Row by numeric index
        actual_index = row_index

    
    ground_truth = sample_df.iloc[:, :output_dim].values  # shape: (1, output_dim)

    # 3) Scale the input columns for the model
    row_input = sample_df.iloc[:, output_dim : output_dim + input_dim].values
    scaled_row_input = input_scaler.transform(row_input)  # shape: (1, input_dim)

    # 4) Run the model
    input_tensor = torch.tensor(scaled_row_input, dtype=torch.float32)
    with torch.no_grad():
        _, reconstruction = model(input_tensor)  # shape: (1, output_dim)

    # 5) Inverse-scale reconstruction so it's comparable to ground truth in original scale
    reconstruction_np = reconstruction.cpu().numpy()  # still scaled, shape: (1, output_dim)
    reconstruction_unscaled = output_scaler.inverse_transform(reconstruction_np)

    # 6) Print results
    print("Row Index Used:", actual_index)
    print("Ground Truth (Original Scale):\n", ground_truth[0][:7].round(4))  
    print("Reconstructed Output (Original Scale):\n", reconstruction_unscaled[0][:7].round(4))  

    # (7) Plot the remaining columns as 24 (x,y,z) points
    #     - from column 7 onward, we have 48 values (24 * 2) = 24 sets of (x,y)
    ground_truth_xyz = ground_truth[0][7:].reshape(-1, 2)
    reconstruction_xyz = reconstruction_unscaled[0][7:].reshape(-1, 2)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot()

    # Plot ground-truth points
    ax.scatter(
        ground_truth_xyz[:, 0], 
        ground_truth_xyz[:, 1], 
        label="Ground Truth"
    )

    # Plot reconstructed points
    ax.scatter(
        reconstruction_xyz[:, 0], 
        reconstruction_xyz[:, 1], 
        label="Reconstruction"
    )

    # Add axis labels, legend, and show
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.show()

def main():

    # Use this block only if you want to load and test an already trained model:
    
    df = load_csv("final_data.csv")
    loaded_input_scaler, loaded_output_scaler = load_scalers("/home/ubuntu/Mojograsp/mojo-grasp/demos/rl_demo/input_scaler.pkl", "/home/ubuntu/Mojograsp/mojo-grasp/demos/rl_demo/output_scaler.pkl")
    loaded_model = load_trained_model("/home/ubuntu/Mojograsp/mojo-grasp/test_best_autoencoder_16.pth", 72, 16, 55)
    
    # Test a single (random) row from the original dataframe multiple times
    # for i in range(10):
    test_single_row(loaded_model, df, loaded_input_scaler, loaded_output_scaler,72, 55)
    
    """

    # This block trains a new model using Weighted MSE.
    config = init_wandb()
    df = load_csv("final_data.csv")
    
    # Preprocess data
    train_dataset, test_dataset, input_scaler, output_scaler = preprocess_data(
        df, config.input_dim, config.output_dim
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create autoencoder model
    model = Autoencoder(config.input_dim, config.latent_dim, config.output_dim)
    save_model(model, "untrained_autoencoder.pth")
    
    # Save input and output scalers
    save_scalers(input_scaler, output_scaler)

    # ---------------------------
    # Define your weighting here:
    # ---------------------------
    weights = torch.ones(config.output_dim, dtype=torch.float32)
    weights[:4] = 8.0
    weights[4:7] = 20.0

    # Create a Weighted MSE Loss function
    criterion = WeightedMSELoss(weights)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Watch model in wandb
    wandb.watch(model, log="all")

    # Train with Weighted MSE
    train_model(model, train_loader, test_loader, criterion, optimizer, config)
    
    wandb.finish()
    print("Training complete with Weighted MSELoss!")
    """

# if __name__ == "__main__":
#     main()





'''
    def __init__(self, input_dim, latent_dim, output_dim):
        super(ImprovedAutoencoder, self).__init__()

        # Encoder: deeper with BatchNorm
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, latent_dim),
            nn.Tanh()  # Better for latent space
        )

        # Decoder: symmetric to encoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed
'''
