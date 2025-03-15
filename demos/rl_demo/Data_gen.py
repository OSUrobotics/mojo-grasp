import numpy as np
import pandas as pd
import random
import json
from scipy.spatial.transform import Rotation as R

def generate_random_poses(
    num_samples,
    x_range=(-0.1, 0.1),
    y_range=(0.00, 0.2),
    z_range=(0.045, 0.055),
    x_rot_range=(-np.pi/8, np.pi/8),
    y_rot_range=(-np.pi/8, np.pi/8),
    z_rot_range=(0, 2 * np.pi)
):
    """
    Generate random x, y, z along with random Euler angles (xyz) 
    converted to quaternions, and return a DataFrame.
    """
    # 1) Generate x, y, z
    x_vals = np.random.uniform(*x_range, num_samples)
    y_vals = np.random.uniform(*y_range, num_samples)
    z_vals = np.random.uniform(*z_range, num_samples)
    
    # 2) Generate Euler angles
    x_euler = np.random.uniform(*x_rot_range, num_samples)
    y_euler = np.random.uniform(*y_rot_range, num_samples)
    z_euler = np.random.uniform(*z_rot_range, num_samples)
    euler_angles = np.vstack((x_euler, y_euler, z_euler)).T
    
    # Convert Euler angles to quaternions
    quaternions = R.from_euler('xyz', euler_angles).as_quat()  # shape (num_samples, 4)
    
    # Create a DataFrame
    df = pd.DataFrame({
        "x": x_vals,
        "y": y_vals,
        "z": z_vals,
        "qx": quaternions[:, 0],
        "qy": quaternions[:, 1],
        "qz": quaternions[:, 2],
        "qw": quaternions[:, 3]
    })
    
    return df

def append_random_shape_slices(df, slices_json_path):
    """
    For each row in `df`, pick a random slice from slices.json,
    which is 24×3, then either:
      - zero out the third column, or
      - just take the first two columns.
    
    Flatten it (24×2 -> 48 columns) and append to the DataFrame.
    """
    # Read the slices JSON
    with open(slices_json_path, "r") as f:
        obj_dict = json.load(f)
    
    shape_keys = list(obj_dict.keys())
    
    # We'll store each row's flattened shape in a list of arrays
    all_flat_shapes = []
    
    for _ in range(len(df)):
        random_key = random.choice(shape_keys)
        # shape_points is an array of shape (24, 3)
        shape_points = np.array(obj_dict[random_key], dtype=float)  
        
        # Option A: Simply ignore the third column:
        shape_2d = shape_points[:, :2]

        # Option B: Force it to 0 explicitly, then slice out the first two columns:
        # shape_points[:, 2] = 0.0
        # shape_2d = shape_points[:, :2]
        
        # Flatten to length 48
        flat_shape = shape_2d.flatten()
        all_flat_shapes.append(flat_shape)
    
    # Convert list of flattened shapes to a DataFrame
    shape_cols = [f"shape_{i}" for i in range(48)]
    shape_df = pd.DataFrame(all_flat_shapes, columns=shape_cols)
    
    # Concatenate with the original df
    df_out = pd.concat([df.reset_index(drop=True), shape_df], axis=1)
    return df_out

def get_dynamic(shape_2d, pose, orientation):
    """
    Computes the dynamic state of the object by applying a quaternion rotation 
    and translation to the input shape (24×2).
    
    1) Expand to 24×3 by adding z=0
    2) Rotate using quaternion
    3) Translate
    4) Return shape with size 24×3
    """
    # Expand from (24,2) to (24,3) by appending z=0
    shape_3d = np.hstack((shape_2d, np.full((shape_2d.shape[0], 1), 0.0)))
    
    x, y, z = pose
    qx, qy, qz, qw = orientation

    # Create the rotation matrix
    rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()  # 3×3
    
    # Rotate the shape
    shape_3d = shape_3d @ rotation_matrix.T
    
    # Translate
    shape_3d[:, 0] += x
    shape_3d[:, 1] += y
    shape_3d[:, 2] += z
    
    return shape_3d  # shape (24,3)

def apply_dynamic_transformation(df):
    """
    For each row, reconstruct the 24×2 shape from columns shape_0...shape_47,
    apply get_dynamic(...), flatten the 24×3 back to 72 columns,
    and append to the DataFrame.
    """
    # Identify the shape columns
    shape_cols = [f"shape_{i}" for i in range(48)]
    dynamic_cols = [f"dyn_{i}" for i in range(72)]
    
    # We'll hold all of the dynamic shapes row-by-row here
    dynamic_data = []
    
    for _, row in df.iterrows():
        # Rebuild the 24×2 shape
        flat_shape = row[shape_cols].to_numpy()  # length 48
        shape_2d = flat_shape.reshape(24, 2)
        
        # Extract pose, orientation
        pose = (row["x"], row["y"], row["z"])
        orientation = (row["qx"], row["qy"], row["qz"], row["qw"])
        
        # Transform
        shape_3d = get_dynamic(shape_2d, pose, orientation)  # 24×3
        
        # Flatten 24×3 -> 72
        flat_3d = shape_3d.flatten()
        dynamic_data.append(flat_3d)
    
    # Build DataFrame for dynamic data
    dynamic_df = pd.DataFrame(dynamic_data, columns=dynamic_cols)
    
    # Concatenate original df with new dynamic columns
    df_out = pd.concat([df, dynamic_df], axis=1)
    return df_out

def main():
    # Step 1 & 2: Generate random poses
    df_poses = generate_random_poses(num_samples=500_000)  
    
    # Step 3: Append a random shape slice (24×3) 
    # but only keep x,y then flatten to 48 columns
    df_with_shapes = append_random_shape_slices(df_poses, "slices.json")
    
    print("DataFrame shape after appending shape slices (should be 7+48=55 columns):",
          df_with_shapes.shape)
    
    # Step 4: Apply dynamic transformation -> 24×3 => flatten to 72
    df_final = apply_dynamic_transformation(df_with_shapes)
    
    # Step 5: Check final shape
    # Now we have 7 + 48 + 72 = 127 columns
    print("Final DataFrame shape (expected 127 columns):", df_final.shape)
    
    # Step 6: Save to CSV
    df_final.to_csv("final_data.csv", index=False)
    print("Saved final_data.csv with dynamic columns included.")

if __name__ == "__main__":
    main()
