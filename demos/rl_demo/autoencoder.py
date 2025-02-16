import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def init_wandb():
    wandb.init(
        project="autoencoder-project",
        config={
            "input_dim": 72,
            "latent_dim": 16,
            "output_dim": 54,
            "learning_rate": 0.001,
            "epochs": 10,
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
            nn.Linear(128, latent_dim),
            nn.Sigmoid()  # Normalizing latent space
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()  # Normalizing outputs
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

def save_model(model, filename="untrained_autoencoder.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")

def load_csv(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, input_dim, output_dim, test_size=0.25):
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    data_x = df_normalized.iloc[:, :input_dim].values
    data_y = df_normalized.iloc[:, 4:4+output_dim].values
    
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_size, random_state=42)
    return TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), \
           TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

def train_model(model, data_loader, criterion, optimizer, config):
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        for batch_x, batch_y in data_loader:
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

def load_trained_model(model_path, input_dim, latent_dim, output_dim):
    model = Autoencoder(input_dim, latent_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  
    return model

def main():
    config = init_wandb()
    # df = load_csv("data.csv")  # Replace with actual file path
    # train_dataset, test_dataset = preprocess_data(df, config.input_dim, config.output_dim)
    # data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # model = Autoencoder(config.input_dim, config.latent_dim, config.output_dim)
    # save_model(model)  # Save before training
    
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # wandb.watch(model, log="all")
    
    # train_model(model, data_loader, criterion, optimizer, config)
    
    # wandb.finish()
    # print("Training complete!")
    model = Autoencoder(config.input_dim, config.latent_dim, config.output_dim)
    save_model(model, "untrained_autoencoder.pth")  # Save untrained model

if __name__ == "__main__":
    main()
