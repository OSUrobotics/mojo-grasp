import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle 
import numpy as np
def init_wandb():
    wandb.init(
        project="autoencoder-project",
        config={
            "input_dim": 72,
            "latent_dim": 16,
            "output_dim": 54,
            "learning_rate": 0.001,
            "epochs": 100,
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
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
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
    X_train, X_test, y_train, y_test = train_test_split(scaled_x, scaled_y, test_size=test_size, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.float32))
    
    # Return both datasets and the two scalers
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
            latent, reconstructed = model(batch_x)
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
            torch.save(model.state_dict(), "best_autoencoder_16.pth")
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

def test_single_row(model, df, input_scaler, input_dim, output_scaler, output_dim, row_index=None):
    """
    Tests a single row from the dataframe:
    - If row_index is None, a random row is chosen.
    - The row is scaled using the provided input_scaler.
    - The appropriate input slice is passed through the model.
    - Both latent and reconstructed outputs are printed.
    """
    # Choose a random row if no index is provided
    if row_index is None:
        sample_df = df.sample(1)
    else:
        sample_df = df.iloc[[row_index]]  # Keep 2D structure
    
    # Extract only the input columns and scale them with the input_scaler
    sample_input = sample_df.iloc[:, output_dim:output_dim+input_dim]
    scaled_sample_input = input_scaler.transform(sample_input)
    
    sample_tensor = torch.tensor(scaled_sample_input, dtype=torch.float32)
    latent, reconstruction = model(sample_tensor)
    # print("Latent Representation:", latent)
    # Optionally, you can print the reconstructed output:
    # print("Reconstructed Output:", reconstruction)
    autoencoder_output = output_scaler.inverse_transform(reconstruction.detach())
    print('autoencoder position',autoencoder_output[0][0:3])
    print('recon position      ', reconstruction[0][0:3])
    print('pickle file position', sample_df['x'],sample_df['y'],sample_df['z'])
    print('autoencoder orientation',autoencoder_output[0][3:7])
    print('recon orientation      ', reconstruction[0][3:7])
    print('pickle file orientation', sample_df['qx'], sample_df['qy'], sample_df['qz'], sample_df['qw'])

def test_pickle_file(model, pkl_path, input_scaler, output_scaler):
    with open(pkl_path,'rb') as file:
        data = pickle.load(file)
    
    data = data['timestep_list']
    for state_stuff in data:
        
        state_stuff = state_stuff['state']
        sample_input = state_stuff['dynamic']
        # sample_input[:,2] += 0.05

        print(sample_input)
        print(state_stuff['obj_2']['pose'])
        scaled_sample_input = input_scaler.transform(np.reshape(sample_input,(1,72)))
        
        sample_tensor = torch.tensor(scaled_sample_input, dtype=torch.float32)
        latent, reconstruction = model(sample_tensor)
        print("reconstruction:", np.shape(reconstruction))
        autoencoder_output = output_scaler.inverse_transform(reconstruction.detach())
        print(np.shape(autoencoder_output))
        print('autoencoder position',autoencoder_output[0][0:3])
        # print('recon position      ', reconstruction[0][0:3])
        print('pickle file position',state_stuff['obj_2']['pose'][0])
        print('autoencoder orientation',autoencoder_output[0][3:7])
        # print('recon orientation      ', reconstruction[0][3:7])
        print('pickle file orientation',state_stuff['obj_2']['pose'][1])
        input('go?')

def main():
    """
    # Load CSV file
    df = load_csv("final_data.csv")
    
    # Preprocess data and get the fitted scalers
    train_dataset, test_dataset, input_scaler, output_scaler = preprocess_data(df, 72, 54)
    
    # Save the scalers using pickle
    save_scalers(input_scaler, output_scaler, "input_scaler.pkl", "output_scaler.pkl")
    
    # Load the scalers (demonstration of the load function)
    loaded_input_scaler, loaded_output_scaler = load_scalers("input_scaler.pkl", "output_scaler.pkl")
    
    # Load the trained autoencoder model
    loaded_model = load_trained_model("/home/ubuntu/Mojograsp/mojo-grasp/demos/rl_demo/best_autoencoder_16.pth", 72, 16, 54)
    
    # Test a single (random) row from the original dataframe multiple times
    for i in range(10):
        test_single_row(loaded_model, df, loaded_input_scaler, 72, 54)

    # Uncomment the following block to run training instead:
    """
    # config = init_wandb()
    # df = load_csv("final_data.csv")
    
    # train_dataset, test_dataset, input_scaler, output_scaler = preprocess_data(df, config.input_dim, config.output_dim)
    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # model = Autoencoder(config.input_dim, config.latent_dim, config.output_dim)
    # save_model(model)  # Save the untrained model if desired
    # save_scalers(input_scaler, output_scaler)
    
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # wandb.watch(model, log="all")
    
    # train_model(model, train_loader, test_loader, criterion, optimizer, config)
    
    # wandb.finish()
    # print("Training complete!")
    config={
            "input_dim": 72,
            "latent_dim": 16,
            "output_dim": 54,
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 64,
        }
    model = load_trained_model('./best_autoencoder_16.pth',config['input_dim'], config['latent_dim'], config['output_dim'])
    input_scalar, output_scalar = load_scalers('scaler.pkl')
    test_pickle_file(model, './data/Static_2/square_A/Episode_775.pkl', input_scalar, output_scalar)
    # test_single_row(model, df, input_scalar, config['input_dim'], output_scalar, config['output_dim'])

if __name__ == "__main__":
    main()
