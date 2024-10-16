import numpy as np
import trimesh
import matplotlib.pyplot as plt

def get_sliced_points(obj_file, y_layer):
    # Load the .obj file
    mesh = trimesh.load_mesh(obj_file)

    # Slice the mesh at the given y layer
    sliced_mesh = mesh.section(plane_origin=[0, y_layer, 0], plane_normal=[0, 1, 0])

    # Check if the slicing was successful
    if sliced_mesh is None:
        print("Slicing failed. No intersection found at the given y layer.")
        return None

    # Get the vertices of the sliced mesh
    slice_2D, _ = sliced_mesh.to_planar()
    vertices = slice_2D.vertices

    theta = np.deg2rad(90)  # Convert 20 degrees to radians
    rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)], 
    [np.sin(theta), np.cos(theta)]])
    rotated_vertices = vertices @ rotation_matrix.T  # Apply rotation matrix

    return rotated_vertices

def weighted_distance(last_vertex, candidate_vertex, previous_vertex, weight_factor):
    # Calculate Euclidean distance
    euclidean_distance = np.linalg.norm(candidate_vertex - last_vertex)
    
    # Calculate perpendicular distance to the line segment formed by last_vertex and previous_vertex
    if previous_vertex is not None:
        line_vec = last_vertex - previous_vertex
        point_vec = candidate_vertex - last_vertex
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        point_vec_scaled = point_vec / line_len
        t = np.dot(line_unitvec, point_vec_scaled)
        nearest = line_vec * t
        perpendicular_distance = np.linalg.norm(nearest - point_vec)
    else:
        perpendicular_distance = 0

    # Combine distances with the weight factor
    return euclidean_distance + weight_factor * perpendicular_distance

def connect_vertices(vertices, weight_factor=0.5):
    if vertices is None or len(vertices) < 2:
        print("No vertices to connect.")
        return None

    # Calculate the centroid of the vertices
    centroid = np.mean(vertices, axis=0)

    # Sort vertices by angle from the centroid
    angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_vertices = vertices[sorted_indices]

    # Initialize the list of connected vertices
    connected_vertices = [sorted_vertices[0]]
    unconnected_vertices = list(sorted_vertices[1:])
    
    # Connect vertices by taking them in sorted order
    for vertex in unconnected_vertices:
        connected_vertices.append(vertex)

    return np.array(connected_vertices)

def get_intersection_points(vertices, center=(0,0), angle_step=15):
    intersections = []  # To store intersection points

    # Generate radial lines every angle_step degrees
    for angle in range(0, 360, angle_step):
        radians = np.deg2rad(angle)
        x_end = (center[0] + np.cos(radians)) * 0.05
        y_end = (center[1] + np.sin(radians)) * 0.05

        # Check for intersection with each line segment of the outline
        for i in range(len(vertices)):
            # Connect the last segment to the first segment for the closed loop
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]  # Wrap around to the first vertex
            intersection = get_line_intersection((center[0], center[1]), (x_end, y_end), p1, p2)
            if intersection is not None:
                intersections.append(intersection)

    return np.array(intersections)

def plot_connected_vertices(vertices, intersections):
    if vertices is None:
        print("No vertices to plot.")
        return

    # Create a 2D plot
    fig, ax = plt.subplots()

    # Plot the connected vertices (outline of the shape)
    ax.plot(vertices[:, 0], vertices[:, 1], label='Outline', color='blue')
    ax.plot([vertices[-1, 0], vertices[0, 0]], [vertices[-1, 1], vertices[0, 1]], color='blue')

    # Plot intersection points
    if intersections.size > 0:
        ax.plot(intersections[:, 0], intersections[:, 1], 'go')  # 'go' for green points

    # Set the plot limits and labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.axis('equal')
    plt.title('State 2')
    plt.legend()
    plt.show()

def get_line_intersection(p1, p2, p3, p4):
    """
    Calculate the intersection point of two lines (p1 to p2 and p3 to p4).
    Returns the intersection point (x, y) if it exists, else None.
    """
    # Convert points to numpy arrays for easy calculations
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    p4 = np.array(p4)

    # Calculate the direction vectors of the lines
    d1 = p2 - p1
    d2 = p4 - p3

    # Solve for t and u parameters
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if denom == 0:
        return None  # Lines are parallel

    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        # Calculate the intersection point
        intersection = p1 + t * d1
        return intersection
    return None  # No intersection within the segments

# Example usage
obj_file = '/home/ubuntu/Mojograsp/mojo-grasp/demos/rl_demo/resources/object_models/Jeremiah_Shapes/Shapes/40x40_triangle.obj'
y_layer = 0.05
vertices = get_sliced_points(obj_file, y_layer)
weight_factor = 0.4  # Adjust this value as needed
connected_vertices = connect_vertices(vertices, weight_factor)

# Get intersection points
center = (0, 0)  # Adjust based on your plot's center
intersections = get_intersection_points(connected_vertices)
print(intersections)

# Plot connected vertices and intersection points
plot_connected_vertices(connected_vertices, intersections)
