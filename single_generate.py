from dataset_generator.generator import generate_lbm_image
from dataset_generator.sdf import generate_sdf_image
from dataset_generator.pure_image import generate_pure_image
import numpy as np
import csv
import os
from tqdm import tqdm
import time

Point = [0.3, 1]
VectorA = [1.4, 0.9]
VectorB = [0.9, -0.8]
fileName = '00002'

if __name__ == "__main__":

    x_min, x_max = 0, 4
    y_min, y_max = 0, 2
    total_iterations = 8000
    start_time = time.time()


    filename = "special"
    print(f"Generating {filename}...")

    generate_lbm_image(os.path.join("images",filename), Point, VectorA, VectorB, 200)
    generate_sdf_image(os.path.join("images",  filename), Point, VectorA, VectorB)
    generate_pure_image(os.path.join("images", filename), Point, VectorA, VectorB)


