import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon

def generate_sdf_image(image_number,point,vectorA,vectorB):
    def point_in_parallelogram(point, vertex1, vertex2, vertex3, vertex4):
        # Check if the point lies within the parallelogram
        edges = [(vertex1, vertex2), (vertex2, vertex3), (vertex3, vertex4), (vertex4, vertex1)]
        for edge_start, edge_end in edges:
            if is_point_on_line_segment(point, edge_start, edge_end):
                return 0  # Point is on the quadrilateral boundary
        # Calculate vectors from vertices to the point
        vectors_to_point = [point - vertex for vertex in [vertex1, vertex2, vertex3, vertex4]]

        # Check if the point is on the same side of all edges
        for i in range(len(edges)):
            edge_start, edge_end = edges[i]
            next_edge_start, next_edge_end = edges[(i + 1) % len(edges)]
            edge_normal = np.array([edge_end[1] - edge_start[1], -(edge_end[0] - edge_start[0])])
            next_edge_normal = np.array(
                [next_edge_end[1] - next_edge_start[1], -(next_edge_end[0] - next_edge_start[0])])

            if np.dot(vectors_to_point[i], edge_normal) * np.dot(vectors_to_point[(i + 1) % len(edges)],
                                                                 next_edge_normal) <= 0:
                return 1  # Point is outside the quadrilateral

        return -1  # Point is inside the quadrilateral
    def is_point_on_line_segment(point, line_start, line_end):
        # Check if the point lies on the line segment defined by line_start and line_end
        vec_start_to_point = point - line_start
        vec_start_to_end = line_end - line_start
        projection_length = np.dot(vec_start_to_point, vec_start_to_end) / np.dot(vec_start_to_end, vec_start_to_end)
        if 0 <= projection_length <= 1:
            projected_point = line_start + projection_length * vec_start_to_end
            return np.allclose(point, projected_point)
        else:
            return False


    def distance_to_line(point, line_start, line_end):
        # Calculate vector from line start to point and from line start to line end
        vec_point_to_start = point - line_start
        vec_line = line_end - line_start

        # Calculate the projection of the point onto the line
        projection_length = np.dot(vec_point_to_start, vec_line) / np.dot(vec_line, vec_line)

        # Check if the projection falls within the line segment
        if 0 <= projection_length <= 1:
            # Calculate the point on the line closest to the given point
            closest_point_on_line = line_start + projection_length * vec_line
            # Calculate the distance between the given point and the closest point on the line
            distance = np.linalg.norm(point - closest_point_on_line)
        else:
            # If the projection falls outside the line segment, calculate distance to nearest endpoint
            distance_to_start = np.linalg.norm(point - line_start)
            distance_to_end = np.linalg.norm(point - line_end)
            # Choose the shorter distance
            distance = min(distance_to_start, distance_to_end)

        return distance
    def signed_distance_function(x, y, origin, vector_a, vector_b):
        # Calculate the distance from the point (x, y) to the parallelogram
        point = origin
        point_2 = origin + vector_a
        point_3 = origin + vector_a + vector_b
        point_4 = origin + vector_b

        sign = point_in_parallelogram(np.array([x,y]),point,point_2,point_3,point_4)
        # if (x==49 and y ==49):
        #     print("0")
        distance =  min(distance_to_line(np.array([x, y]), point, point_2),
                        distance_to_line(np.array([x, y]), point_2, point_3),
                        distance_to_line(np.array([x, y]), point_3, point_4),
                        distance_to_line(np.array([x, y]), point_4, point),
                        )
        # if sign==-1:
        #     print(f'x={x} y={y} sign={sign} distance={distance}')
        return sign * distance

    def generate_sdf_image(width, height, origin, vector_a, vector_b):
        sdf_image = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                sdf_image[y, x] = signed_distance_function(x, y, origin, vector_a, vector_b)
        return sdf_image

    # Define parallelogram parameters
    origin = np.array(point)*64  # Origin point of the parallelogram
    vector_a = np.array(vectorA)*64 # VectorA
    vector_b = np.array(vectorB)*64 # VectorB

    # Generate SDF image
    width = 256
    height = 128
    sdf_image = generate_sdf_image(width, height, origin, vector_a, vector_b)

    # Normalize SDF values for visualization (optional)
    # sdf_image /= max(np.max(sdf_image), -np.min(sdf_image))
    # print(np.max(sdf_image))
    # print(np.min(sdf_image))
    # Display or save the SDF image
    # Example:

    colors = [(0.5, 0, 1), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]  # 紫到红
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    vertex1 = origin
    vertex2 = origin + vector_a
    vertex3 = origin + vector_a + vector_b
    vertex4 = origin + vector_b
    plt.figure()
    # plt.figure(figsize=(width / 100, height / 100), dpi=100)  # Set the figure size
    plt.imshow(sdf_image, cmap=cmap, origin='lower')
    plt.gca().add_patch(Polygon([vertex1, vertex2, vertex3, vertex4], closed=True, edgecolor='white', facecolor='none'))
    # 关闭坐标轴刻度
    # plt.axis('off')
    plt.colorbar()
    # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)  # Remove whitespace around the image

    picture_name = f"{image_number}_sdf.png"
    plt.savefig(picture_name, bbox_inches='tight', pad_inches=0,dpi=100)
    plt.close()