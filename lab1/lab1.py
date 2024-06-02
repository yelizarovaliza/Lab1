import numpy as np
import matplotlib.pyplot as plt

# Трикутник
triangle = np.array([[0, 0], [1, 0], [0.5, 1], [0, 0]])

# Неправильний багатокутник
polygon = np.array([[0, 0], [1, 0.2], [0.4, 1], [0.5, 0.4], [0, 0.8], [-0.5, 0.4], [-0.4, 1], [-1, 0.2], [0, 0]])

def plot_shape(shape, title="Shape"):
    plt.figure()
    plt.plot(shape[:, 0], shape[:, 1], 'o-', linewidth=2)
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# Візуалізуємо початкові об'єкти
plot_shape(triangle, "Initial Triangle")
plot_shape(polygon, "Initial Polygon")

def rotate(shape, angle):
    radians = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians), np.cos(radians)]
    ])
    return shape @ rotation_matrix.T

rotated_triangle = rotate(triangle, 45)
plot_shape(rotated_triangle, "Rotated Triangle by 45 degrees")
rotated_poygon = rotate(polygon, 30)
plot_shape(rotated_poygon, "Rotated Polygon by 30 degrees")

