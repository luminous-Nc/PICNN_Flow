from dataset_generator.generator import generate_lbm_image
from dataset_generator.sdf import generate_sdf_image
from dataset_generator.pure_image import generate_pure_image
import numpy as np
import csv
import os
from tqdm import tqdm
import time

# Point = [0.3, 0.2]
# VectorA = [1.4, 0.5]
# VectorB = [0.8, -0.1]
# fileName = '00002'

def generate_random_parallelogram():
    while True:
        point_a = (np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max))
        vector_a = (np.random.uniform(-2, 2), np.random.uniform(-2, 2))
        vector_b = (np.random.uniform(-2, 2), np.random.uniform(-2, 2))

        # 计算平行四边形的四个点
        point_b = (point_a[0] + vector_a[0], point_a[1] + vector_a[1])
        point_c = (point_b[0] + vector_b[0], point_b[1] + vector_b[1])
        point_d = (point_a[0] + vector_b[0], point_a[1] + vector_b[1])

        # 检查四个点是否都在范围内
        if (x_min <= point_a[0] <= x_max and y_min <= point_a[1] <= y_max and
                x_min <= point_b[0] <= x_max and y_min <= point_b[1] <= y_max and
                x_min <= point_c[0] <= x_max and y_min <= point_c[1] <= y_max and
                x_min <= point_d[0] <= x_max and y_min <= point_d[1] <= y_max):
            return point_a, vector_a, vector_b


if __name__ == "__main__":
    csv_file = 'parallelogram_data.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Number', 'Point', 'Vector A', 'Vector B'])

    x_min, x_max = 0, 4
    y_min, y_max = 0, 2
    total_iterations = 8000
    start_time = time.time()

    for i in tqdm(range(total_iterations), position=0, desc='Overall Progress'):

        filename = f'{i + 1:05d}'
        print(f"Generating {filename}...")
        Point, VectorA, VectorB = generate_random_parallelogram()
        generate_lbm_image(os.path.join("images", "lbm", filename), Point, VectorA, VectorB, 100)
        generate_sdf_image(os.path.join("images", "sdf", filename), Point, VectorA, VectorB)
        generate_pure_image(os.path.join("images", "pure", filename), Point, VectorA, VectorB)

        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([filename, Point, VectorA, VectorB])


