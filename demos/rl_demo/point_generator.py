import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# Function to extract mesh file and scale from the URDF file
def get_mesh_from_urdf(urdf_file):
    # Parse the URDF XML file
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    # Find the first link with a visual element
    for link in root.findall('link'):
        for visual in link.findall('visual'):
            geometry = visual.find('geometry')
            if geometry is not None:
                mesh = geometry.find('mesh')
                if mesh is not None:
                    # Extract the filename and scale
                    mesh_file = mesh.get('filename')
                    scale_str = mesh.get('scale')

                    if scale_str is not None:
                        scale = np.array([float(s) for s in scale_str.split()])
                    else:
                        scale = np.array([1.0, 1.0, 1.0])  # Default scale

                    # Get the directory of the URDF file and combine it with the mesh filename
                    urdf_dir = os.path.dirname(urdf_file)  # Get the URDF directory
                    full_mesh_path = os.path.join(urdf_dir, mesh_file)  # Append mesh filename to URDF path

                    return full_mesh_path, scale

    print("No mesh found in the URDF.")
    return None, None

# Function to slice the mesh and return scaled, rotated vertices
def get_sliced_points(mesh_file, y_layer, scale):
    # Load the mesh file (OBJ/STL)
    mesh = trimesh.load_mesh(mesh_file)

    # Slice the mesh at the given y layer
    sliced_mesh = mesh.section(plane_origin=[0, y_layer, 0], plane_normal=[0, 1, 0])

    if sliced_mesh is None:
        print("Slicing failed. No intersection found at the given y layer.")
        return None

    # Get the vertices of the sliced mesh
    slice_2D, _ = sliced_mesh.to_planar()
    vertices = slice_2D.vertices

    # Apply the scale to the vertices directly
    scaled_vertices = vertices * scale[:2]  # Only apply the x and z scaling (ignore y)

    # Rotate the vertices by 90 degrees
    theta = np.deg2rad(90)  # Rotation angle (adjust as needed)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)], 
        [np.sin(theta), np.cos(theta)]])
    rotated_vertices = scaled_vertices @ rotation_matrix.T  # Apply rotation matrix

    return rotated_vertices

# Function to connect vertices in a sorted order
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

# Function to find intersection points based on radial lines
def get_intersection_points(vertices, center=(0, 0), angle_step=15):
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

# Function to calculate intersection between two line segments
def get_line_intersection(p1, p2, p3, p4):
    """
    Calculate the intersection point of two lines (p1 to p2 and p3 to p4).
    Returns the intersection point (x, y) if it exists, else None.
    """
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

# Function to plot connected vertices and intersection points
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

def get_slice(urdf_file):
    mesh_file, scale = get_mesh_from_urdf(urdf_file)

    if mesh_file:
        y_layer = 0.05
        vertices = get_sliced_points(mesh_file, y_layer, scale)
        connected_vertices = connect_vertices(vertices)

        intersections = get_intersection_points(connected_vertices).round(6)
        intersections_flattened_rounded = intersections.flatten()

    return intersections
    

# # Example usage with a URDF file
# urdf_file = '/home/ubuntu/Mojograsp/mojo-grasp/demos/rl_demo/resources/object_models/Jeremiah_Shapes/large_40x40_triangle.urdf'
# mesh_file, scale = get_mesh_from_urdf(urdf_file)

# if mesh_file:
#     y_layer = 0.05
#     vertices = get_sliced_points(mesh_file, y_layer, scale)
#     connected_vertices = connect_vertices(vertices)

#     intersections = get_intersection_points(connected_vertices)
#     print(intersections)

#     # Plot connected vertices and intersection points
#     plot_connected_vertices(connected_vertices, intersections)

#(get_slice('/home/ubuntu/Mojograsp/mojo-grasp/demos/rl_demo/resources/object_models/2v2_mod/2v2_mod_cuboid_small.urdf'))
