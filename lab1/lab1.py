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

#rotation
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

# scaling
def scale(shape, scale_factor):
    scaling_matrix = np.array([
        [scale_factor, 0],
        [0, scale_factor]
    ])
    return shape @ scaling_matrix.T

scaled_triangle = scale(triangle, 2)
plot_shape(scaled_triangle, "Scaled Triangle by factor 2")
scaled_polygon = scale(polygon, 0.5)
plot_shape(scaled_polygon, "Scaled Polygon by factor 0.5")

# reflection
def reflect(shape, axis):
    if axis == 'x':
        reflection_matrix = np.array([
            [1, 0],
            [0, -1]
        ])
    elif axis == 'y':
        reflection_matrix = np.array([
            [-1, 0],
            [0, 1]
        ])
    return shape @ reflection_matrix.T

reflected_triangle = reflect(triangle, 'x')
plot_shape(reflected_triangle, "Reflected Triangle over X axis")
reflected_polygon = reflect(polygon, 'y')
plot_shape(reflected_polygon, "Reflected Polygon over Y axis")

# shear
def shear(shape, axis, shear_factor):
    if axis == 'x':
        shear_matrix = np.array([
            [1, shear_factor],
            [0, 1]
        ])
    elif axis == 'y':
        shear_matrix = np.array([
            [1, 0],
            [shear_factor, 1]
        ])
    return shape @ shear_matrix.T

sheared_triangle = shear(triangle, 'x', 1)
plot_shape(sheared_triangle, "Sheared Triangle along X axis with factor 1")
sheared_polygon = shear(polygon, 'y', 3)
plot_shape(sheared_polygon, "Sheared Polygon along Y axis with factor 3")


