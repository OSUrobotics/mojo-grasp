import numpy as np
import trimesh
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString

def slice_obj_at_y_level(obj_file, y_level):
    mesh = trimesh.load(obj_file)
    faces = mesh.faces
    vertices = mesh.vertices
    slice_edges = []

    for face in faces:
        face_vertices = vertices[face]
        face_y = face_vertices[:, 1]

        if (face_y.min() <= y_level <= face_y.max()):
            for i in range(len(face)):
                v0 = face_vertices[i]
                v1 = face_vertices[(i + 1) % len(face)]
                if (v0[1] <= y_level <= v1[1]) or (v1[1] <= y_level <= v0[1]):
                    t = (y_level - v0[1]) / (v1[1] - v0[1])
                    x_intersect = v0[0] + t * (v1[0] - v0[0])
                    z_intersect = v0[2] + t * (v1[2] - v0[2])
                    slice_edges.append([x_intersect, z_intersect])

    return np.array(slice_edges) if slice_edges else None

def slice_obj_at_x_level(obj_file, x_level):
    mesh = trimesh.load(obj_file)
    faces = mesh.faces
    vertices = mesh.vertices
    slice_edges = []

    for face in faces:
        face_vertices = vertices[face]
        face_x = face_vertices[:, 0]

        if (face_x.min() <= x_level <= face_x.max()):
            for i in range(len(face)):
                v0 = face_vertices[i]
                v1 = face_vertices[(i + 1) % len(face)]
                if (v0[0] <= x_level <= v1[0]) or (v1[0] <= x_level <= v0[0]):
                    t = (x_level - v0[0]) / (v1[0] - v0[0])
                    y_intersect = v0[1] + t * (v1[1] - v0[1])
                    z_intersect = v0[2] + t * (v1[2] - v0[2])
                    slice_edges.append([y_intersect, z_intersect])

    return np.array(slice_edges) if slice_edges else None

def slice_obj_at_z_level(obj_file, z_level):
    mesh = trimesh.load(obj_file)
    faces = mesh.faces
    vertices = mesh.vertices
    slice_edges = []

    for face in faces:
        face_vertices = vertices[face]
        face_z = face_vertices[:, 2]

        if (face_z.min() <= z_level <= face_z.max()):
            for i in range(len(face)):
                v0 = face_vertices[i]
                v1 = face_vertices[(i + 1) % len(face)]
                if (v0[2] <= z_level <= v1[2]) or (v1[2] <= z_level <= v0[2]):
                    t = (z_level - v0[2]) / (v1[2] - v0[2])
                    x_intersect = v0[0] + t * (v1[0] - v0[0])
                    y_intersect = v0[1] + t * (v1[1] - v0[1])
                    slice_edges.append([x_intersect, y_intersect])

    return np.array(slice_edges) if slice_edges else None

def calculate_outer_perimeter(points_2d):
    if points_2d is not None:
        polygon = Polygon(points_2d)
        outer_perimeter = polygon.convex_hull
        return outer_perimeter
    return None

def find_intersection_points(outer_perimeter, angle_increment=15):
    intersection_points = []
    
    for angle in range(0, 360, angle_increment):
        radians = np.deg2rad(angle)
        line_end = np.array([np.cos(radians), np.sin(radians)]) * 100  # Extend line to arbitrary length
        line = LineString([np.array([0, 0]), line_end])

        if outer_perimeter.intersects(line):
            intersection_point = outer_perimeter.intersection(line)
            if not intersection_point.is_empty:
                # Append each intersection point, excluding the origin (0, 0)
                if isinstance(intersection_point, LineString):
                    for point in intersection_point.coords:
                        if not np.array_equal(point, (0, 0)):  # Skip the center point
                            intersection_points.append(point)
                else:
                    if not np.array_equal(intersection_point.coords[0], (0, 0)):  # Skip the center point
                        intersection_points.append(intersection_point.coords[0])

    return np.array(intersection_points)

def plot_outer_perimeter(points_2d, outer_perimeter, intersection_points=None):
    if points_2d is not None:
        plt.figure()
        plt.fill(points_2d[:, 0], points_2d[:, 1], alpha=0.5, label='Slice')
        x, y = outer_perimeter.exterior.xy
        plt.plot(x, y, 'r-', linewidth=2, label='Outer Perimeter')

        # Plot intersection points if available
        if intersection_points is not None:
            plt.plot(intersection_points[:, 0], intersection_points[:, 1], 'go', label='Intersections')

        plt.title("2D Slice of OBJ Model with Intersection Points")
        plt.xlabel("X-axis")
        plt.ylabel("Z-axis")
        plt.axis('equal')
        plt.grid()
        plt.legend()
        plt.show()
    else:
        print("No points to plot.")

if __name__ == "__main__":
    obj_file_path = '/home/ubuntu/Mojograsp/mojo-grasp/demos/rl_demo/resources/object_models/2v2_mod/obj_files/2v2_mod_cylinder_small.obj'  # Update with your .obj file path

    # Slice in different directions
    y_slice_level = 0.005
    points_y = slice_obj_at_y_level(obj_file_path, y_slice_level)
    outer_perimeter_y = calculate_outer_perimeter(points_y)
    print(f"Outer Perimeter of the Y-slice: {outer_perimeter_y.length:.2f}")
    intersection_y = find_intersection_points(outer_perimeter_y)
    plot_outer_perimeter(points_y, outer_perimeter_y, intersection_y)
    #print(f"Intersection Points: {intersection_y/2}")
    intersection_points_list = (intersection_y/2).flatten().tolist()
    print(f"Intersection Points: {np.round(intersection_points_list, 6)}")


    # x_slice_level = 0.0
    # points_x = slice_obj_at_x_level(obj_file_path, x_slice_level)
    # outer_perimeter_x = calculate_outer_perimeter(points_x)
    # print(f"Outer Perimeter of the X-slice: {outer_perimeter_x.length:.2f}")
    # intersection_x = find_intersection_points(outer_perimeter_x)
    # plot_outer_perimeter(points_x, outer_perimeter_x, intersection_x)

    # z_slice_level = 0.0
    # points_z = slice_obj_at_z_level(obj_file_path, z_slice_level)
    # outer_perimeter_z = calculate_outer_perimeter(points_z)
    # print(f"Outer Perimeter of the Z-slice: {outer_perimeter_z.length:.2f}")
    # intersection_z = find_intersection_points(outer_perimeter_z)
    # plot_outer_perimeter(points_z, outer_perimeter_z, intersection_z)
