import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
def generate_pure_image(image_number,point,vectorA,vectorB):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # 定义四边形的顶点坐标
    origin = np.array(point)*64
    vector_a = np.array(vectorA)*64
    vector_b = np.array(vectorB)*64

    vertex1 = origin
    vertex2 = [origin[0] + vector_a[0], origin[1] + vector_a[1]]
    vertex3 = [origin[0] + vector_a[0] + vector_b[0], origin[1] + vector_a[1] + vector_b[1]]
    vertex4 = [origin[0] + vector_b[0], origin[1] + vector_b[1]]

    # 创建一个256x128大小的图像
    fig, ax = plt.subplots()
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 128)

    # 绘制四边形
    polygon = patches.Polygon([vertex1, vertex2, vertex3, vertex4], closed=True, edgecolor='black', facecolor='none')
    ax.add_patch(polygon)

    # 显示图像
    plt.gca().set_aspect('equal', adjustable='box')
    picture_name = f"{image_number}_ori.png"
    plt.savefig(picture_name)
