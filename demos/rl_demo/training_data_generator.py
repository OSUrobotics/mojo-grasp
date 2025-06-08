import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Parameters
num_points = 100_000  # Number of points to generate
square_width = 0.75  # Width of the square (meters)
square_height = 0.75  # Height of the square (meters)
center_x = 0.0  # X-coordinate of the center
center_y = 0.1  # Y-coordinate of the center
output_file = "autoencoder_training_data.csv"  # CSV file to save the data

# Calculate bounds based on the center point and dimensions
x_min = center_x - square_width / 2
x_max = center_x + square_width / 2
y_min = center_y - square_height / 2
y_max = center_y + square_height / 2

# Generate random coordinates
x = np.random.uniform(x_min, x_max, num_points)
y = np.random.uniform(y_min, y_max, num_points)

# Generate biased quaternions
def generate_biased_quaternions(num_points, bias_factor=0.1):
    angles = np.random.randn(num_points, 3) * bias_factor  # Small random rotations around x and z
    angles[:, 1] = np.random.uniform(-np.pi, np.pi, size=num_points)  # Full range of rotation around y
    
    quaternions = []
    for angle in angles:
        qx = np.array([np.cos(angle[0] / 2), np.sin(angle[0] / 2), 0, 0])  # Rotation around x
        qy = np.array([np.cos(angle[1] / 2), 0, np.sin(angle[1] / 2), 0])  # Rotation around y
        qz = np.array([np.cos(angle[2] / 2), 0, 0, np.sin(angle[2] / 2)])  # Rotation around z
        
        q = multiply_quaternions(multiply_quaternions(qx, qy), qz)
        quaternions.append(q / np.linalg.norm(q))  # Normalize to ensure validity
    
    return np.array(quaternions)

def multiply_quaternions(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])

quaternions = generate_biased_quaternions(num_points)

# Combine x, y, and quaternions into a DataFrame
data = pd.DataFrame({
    "x": x,
    "y": y,
    "q1": quaternions[:, 0],
    "q2": quaternions[:, 1],
    "q3": quaternions[:, 2],
    "q4": quaternions[:, 3],
})

# Save to CSV
data.to_csv(output_file, index=False)
print(f"Saved {num_points} points with quaternions to {output_file}")

# Plot the x, y points
plt.figure(figsize=(8, 8))
plt.scatter(x, y, s=0.1, alpha=0.5)  # Small marker size for better visualization
plt.title(f"{num_points} Randomly Generated Points in a {square_width}x{square_height} Square (Centered at ({center_x}, {center_y}))")
plt.xlabel("x (meters)")
plt.ylabel("y (meters)")
plt.axis("equal")
plt.show()

# Visualization Function for Quaternions
def visualize_quaternions(quaternions):
    from mpl_toolkits.mplot3d import Axes3D

    # 2D Projection (q1 vs q2)
    plt.figure(figsize=(8, 8))
    plt.scatter(quaternions[:, 0], quaternions[:, 1], s=0.1, alpha=0.5)
    plt.title("Projection of Quaternions onto q1-q2 Plane")
    plt.xlabel("q1")
    plt.ylabel("q2")
    plt.axis("equal")
    plt.show()

    # 3D Projection (q1, q2, q3)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], s=0.1, alpha=0.5
    )
    ax.set_title("Projection of Quaternions onto q1-q2-q3 Space")
    ax.set_xlabel("q1")
    ax.set_ylabel("q2")
    ax.set_zlabel("q3")
    plt.show()

# Visualize Quaternions
#visualize_quaternions(quaternions)


