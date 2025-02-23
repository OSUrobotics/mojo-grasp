import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import json
import random

def generate_euler_angles(x_range, y_range, z_range, num_samples):
    x_angles = np.random.uniform(x_range[0], x_range[1], num_samples)
    y_angles = np.random.uniform(y_range[0], y_range[1], num_samples)
    z_angles = np.random.uniform(z_range[0], z_range[1], num_samples)
    euler_angles = np.vstack((x_angles, y_angles, z_angles)).T
    return euler_angles

# Parameters
num_points = 10000
square_width = 0.1
square_height = 0.1
center_x = 0.0
center_y = 0.1
x_rot_range = (-np.pi/8, np.pi/8)
y_rot_range = (-np.pi/8, np.pi/8)
z_rot_range = (0, 2 * np.pi)
output_file = "flattened_autoencoder_training_data_with_noise.csv"

# Generate random coordinates and Euler angles
x_min = center_x - square_width / 2
x_max = center_x + square_width / 2
y_min = center_y - square_height / 2
y_max = center_y + square_height / 2
x = np.random.uniform(x_min, x_max, num_points)
y = np.random.uniform(y_min, y_max, num_points)
angles = generate_euler_angles(x_rot_range, y_rot_range, z_rot_range, num_points)
quaternions = np.array([R.from_euler('xyz', angle).as_quat() for angle in angles])
#print(quaternions)

rotated_x, rotated_y, rotated_z = [], [], []
for i in range(num_points):
    # Create the point with x and y from the random sample, and 0.05 as the z (height)
    point = np.array([x[i], y[i], 0.05])
    rotation = R.from_quat(quaternions[i])
    # Subtract the base (x, y) to isolate the height offset
    local_point = point - np.array([x[i], y[i], 0])
    rotated_local_point = rotation.apply(local_point)
    # Add the base translation back
    rotated_point = rotated_local_point + np.array([x[i], y[i], 0])
    rotated_x.append(rotated_point[0])
    rotated_y.append(rotated_point[1])
    rotated_z.append(rotated_point[2])

# print(np.max(rotated_x), np.min(rotated_x))
# print(np.max(rotated_y), np.min(rotated_y))
# print(np.max(rotated_z), np.min(rotated_z))

# Create the initial DataFrame
df = pd.DataFrame({
    'x': rotated_x,
    'y': rotated_y,
    'z': rotated_z,
    'qx': quaternions[:, 0],
    'qy': quaternions[:, 1],
    'qz': quaternions[:, 2],
    'qw': quaternions[:, 3]
})

print(df.head())

# Load the JSON file with the object points
with open("slices.json", "r") as f:
    obj_dict = json.load(f)

# Randomly assign an object to each row, flatten its points, and add 10% Gaussian noise
flattened_points_with_noise = []
for _ in range(len(df)):
    obj_key = random.choice(list(obj_dict.keys()))
    obj_points = np.array(obj_dict[obj_key])  # Convert to numpy array
    #noisy_points = obj_points + np.random.normal(0, 0.1 * (np.abs(obj_points)))  # Add 10% Gaussian noise, avoiding zero
    flattened_xy = obj_points[:, :2].flatten()  # Keep only the x and y coordinates
    flattened_points_with_noise.append(flattened_xy)

# Create a DataFrame with flattened noisy x and y points as columns
num_points_per_obj = len(flattened_points_with_noise[0]) // 2
columns = [f'p{i}_{coord}' for i in range(num_points_per_obj) for coord in ['x', 'y']]
points_df = pd.DataFrame(flattened_points_with_noise, columns=columns)

# Concatenate the original DataFrame with the noisy flattened x and y points
final_df = pd.concat([df, points_df], axis=1)

# Print the size of the final DataFrame and save to CSV
#print(f"Final DataFrame size: {final_df.shape}")
#print(final_df.head())
#final_df.to_csv(output_file, index=False)
print(f"Flattened data with noisy x and y points saved to {output_file}")

def manipulate_row(row):
    # print("This is the row", row)
    # Extract quaternion and normalize it
    a, b, c, w = row['qx'], row['qy'], row['qz'], row['qw']
    norm = np.sqrt(a**2 + b**2 + c**2 + w**2)
    a, b, c, w = a / norm, b / norm, c / norm, w / norm
    
    # Construct the 3x3 rotation matrix from the normalized quaternion
    rotation_matrix = np.array([
        [1 - 2 * (b**2 + c**2), 2 * (a * b - w * c),     2 * (a * c + w * b)],
        [2 * (a * b + w * c),     1 - 2 * (a**2 + c**2), 2 * (b * c - w * a)],
        [2 * (a * c - w * b),     2 * (b * c + w * a),   1 - 2 * (a**2 + b**2)]
    ])
    
    # Extract the shape (assuming it's a set of 2D x,y coordinates starting from the 6th column)
    shape = row.iloc[6:54].values.reshape(-1, 2)  # Reshaping into a 2D array of 2 columns (x, y points)
    
    # Append 0.05 as the z value to each (x, y) pair to make (x, y, z) triplets
    shape = np.hstack((shape, np.full((shape.shape[0], 1), 0.05)))  # Adding a z-value of 0.05
    
    # Ensure shape has 24 points (it should have 24 rows, each with 3 columns)
    if shape.shape[0] != 24:
        print("Warning: Expected 24 points in shape but got", shape.shape[0])
    
    # Apply the rotation to each point
    shape_rotated = shape @ rotation_matrix.T  # Perform matrix multiplication
    
    # Apply the translation
    shape_rotated[:, 0] += row['x']
    shape_rotated[:, 1] += row['y']
    shape_rotated[:, 2] += row['z']
    
    # Assign back the transformed points as separate columns for x, y, z
    for i in range(24):  # For each of the 24 points
        row[f'x_{i}'] = shape_rotated[i, 0]
        row[f'y_{i}'] = shape_rotated[i, 1]
        row[f'z_{i}'] = shape_rotated[i, 2]
    
    return row

# Assuming final_df is your DataFrame
final_df = final_df.apply(manipulate_row, axis=1)

# print(final_df.head())

#save the final DataFrame to a CSV file
final_df.to_csv("final_data.csv", index=False)

