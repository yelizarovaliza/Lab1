import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

# 2D shapes
triangle = np.array([[0, 0], [1, 0], [0.5, 1], [0, 0]])
polygon = np.array([[0, 0], [1, 0.2], [0.4, 1], [0.5, 0.4], [0, 0.8], [-0.5, 0.4], [-0.4, 1], [-1, 0.2], [0, 0]])

# 3D shape
tetrahedron = np.array([
    [1, 1, 1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [1, 1, 1]
])

# 2D shapes plot
def plot_shape(shape, title="Shape"):
    plt.figure()
    plt.plot(shape[:, 0], shape[:, 1], 'o-', linewidth=2)
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# 3D shapes plot
def plot_3d_shape(shape, title="3D Shape"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(shape[:, 0], shape[:, 1], shape[:, 2], 'o-', linewidth=2)
    ax.set_title(title)
    plt.show()

# rotation
def rotate(shape, angle):
    radians = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians), np.cos(radians)]
    ])
    return shape @ rotation_matrix.T

# scaling
def scale(shape, scale_factor):
    scaling_matrix = np.array([
        [scale_factor, 0],
        [0, scale_factor]
    ])
    return shape @ scaling_matrix.T

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

# shearing
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

def transform(shape, transformation_matrix):
    return shape @ transformation_matrix.T

# 3D transformation
def rotate_3d(shape, axis, angle):
    radians = np.deg2rad(angle)
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(radians), -np.sin(radians)],
            [0, np.sin(radians), np.cos(radians)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(radians), 0, np.sin(radians)],
            [0, 1, 0],
            [-np.sin(radians), 0, np.cos(radians)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(radians), -np.sin(radians), 0],
            [np.sin(radians), np.cos(radians), 0],
            [0, 0, 1]
        ])
    return shape @ rotation_matrix.T

# custom matrix
def scale_3d(shape, scale_factor):
    scaling_matrix = np.diag([scale_factor, scale_factor, scale_factor])
    return shape @ scaling_matrix.T


# Image transformation
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image

def scale_image(image, scale_factor):
    return cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

def reflect_image(image, axis):
    if axis == 'x':
        return cv2.flip(image, 0)
    elif axis == 'y':
        return cv2.flip(image, 1)

def shear_image(image, shear_factor, axis):
    (h, w) = image.shape[:2]
    if axis == 'x':
        shear_matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    elif axis == 'y':
        shear_matrix = np.float32([[1, 0, 0], [shear_factor, 1, 0]])
    sheared_image = cv2.warpAffine(image, shear_matrix, (w, h))
    return sheared_image

def transform_image(image, matrix):
    (h, w) = image.shape[:2]
    transformed_image = cv2.warpAffine(image, matrix, (w, h))
    return transformed_image

# Load image
image = cv2.imread('image.jpg')

# Menu
def menu():
    while True:
        print("\nChoose an option:")
        print("0. Exit")
        print("1. Show initial shape")
        print("2. Rotate")
        print("3. Scale")
        print("4. Reflect")
        print("5. Shear")
        print("6. Custom Transformation")
        print("7. Rotate 3D (only for 3D figure)")
        print("8. Scale 3D (only for 3D figure)")
        print("9. Image Transformations")
        
        
        choice = input("Enter the number of your choice: ")
        
        if choice == '0':
            break
        
        if choice == '1':
            print("\nChoose a figure to display:")
            print("1. Triangle (2D)")
            print("2. Polygon (2D)")
            print("3. Tetrahedron (3D)")
            
            figure_choice = input("Enter the number of the figure: ")
            
            if figure_choice == '1':
                plot_shape(triangle, "Initial Triangle")
            elif figure_choice == '2':
                plot_shape(polygon, "Initial Polygon")
            elif figure_choice == '3':
                plot_3d_shape(tetrahedron, "Initial Tetrahedron")
            else:
                print("Invalid choice. Please try again.")
        
        elif choice in ['2', '3', '4', '5', '6', '7', '8']:
            print("\nChoose a figure:")
            print("1. Triangle (2D)")
            print("2. Polygon (2D)")
            print("3. Tetrahedron (3D)")

            figure_choice = input("Enter the number of the figure: ")
            
            if figure_choice not in ['1', '2', '3']:
                print("Invalid choice. Please try again.")
                continue
            
            if choice == '2':  # Rotate
                angle = float(input("Enter the angle of rotation: "))
                if figure_choice == '1':
                    result = rotate(triangle, angle)
                    plot_shape(result, f"Rotated Triangle by {angle} degrees")
                elif figure_choice == '2':
                    result = rotate(polygon, angle)
                    plot_shape(result, f"Rotated Polygon by {angle} degrees")
                elif figure_choice == '3':
                    axis = input("Enter the axis of rotation (x, y, or z): ").lower()
                    result = rotate_3d(tetrahedron, axis, angle)
                    plot_3d_shape(result, f"Rotated Tetrahedron by {angle} degrees along {axis} axis")
            
            elif choice == '3':  # Scale
                scale_factor = float(input("Enter the scale factor: "))
                if figure_choice == '1':
                    result = scale(triangle, scale_factor)
                    plot_shape(result, f"Scaled Triangle by factor {scale_factor}")
                elif figure_choice == '2':
                    result = scale(polygon, scale_factor)
                    plot_shape(result, f"Scaled Polygon by factor {scale_factor}")
                elif figure_choice == '3':
                    result = scale_3d(tetrahedron, scale_factor)
                    plot_3d_shape(result, f"Scaled Tetrahedron by factor {scale_factor}")
            
            elif choice == '4':  # Reflect
                axis = input("Enter the axis of reflection (x or y): ").lower()
                if figure_choice == '1':
                    result = reflect(triangle, axis)
                    plot_shape(result, f"Reflected Triangle over {axis.upper()} axis")
                elif figure_choice == '2':
                    result = reflect(polygon, axis)
                    plot_shape(result, f"Reflected Polygon over {axis.upper()} axis")
            
            elif choice == '5':  # Shear
                axis = input("Enter the axis of shear (x or y): ").lower()
                shear_factor = float(input(f"Enter the shear factor along {axis.upper()} axis: "))
                if figure_choice == '1':
                    result = shear(triangle, axis, shear_factor)
                    plot_shape(result, f"Sheared Triangle along {axis.upper()} axis with factor {shear_factor}")
                elif figure_choice == '2':
                    result = shear(polygon, axis, shear_factor)
                    plot_shape(result, f"Sheared Polygon along {axis.upper()} axis with factor {shear_factor}")
            
            elif choice == '6':  # Custom Transformation
                matrix = input("Enter the transformation matrix as comma-separated values (e.g., 1,0,0,1 for 2x2 identity matrix): ")
                matrix = np.array(matrix.split(','), dtype=float).reshape((2, 2))
                if figure_choice == '1':
                    result = transform(triangle, matrix)
                    plot_shape(result, "Transformed Triangle with custom matrix")
                elif figure_choice == '2':
                    result = transform(polygon, matrix)
                    plot_shape(result, "Transformed Polygon with custom matrix")
            
            elif choice == '7':  # Rotate 3D
                if figure_choice == '3':
                    axis = input("Enter the axis of rotation (x, y, or z): ").lower()
                    angle = float(input("Enter the angle of rotation: "))
                    result = rotate_3d(tetrahedron, axis, angle)
                    plot_3d_shape(result, f"Rotated Tetrahedron by {angle} degrees along {axis} axis")
                else:
                    print("Invalid choice. 3D transformations are only available for Tetrahedron.")
            
            elif choice == '8':  # Scale 3D
                if figure_choice == '3':
                    scale_factor = float(input("Enter the scale factor: "))
                    result = scale_3d(tetrahedron, scale_factor)
                    plot_3d_shape(result, f"Scaled Tetrahedron by factor {scale_factor}")
                else:
                    print("Invalid choice. 3D transformations are only available for Tetrahedron.")
        
        elif choice == '9':  # Image Transformations
            print("\nChoose an image transformation:")
            print("1. Rotate")
            print("2. Scale")
            print("3. Reflect")
            print("4. Shear")
            print("5. Custom Transformation")
            
            img_choice = input("Enter the number of the transformation: ")
            
            if img_choice == '1':  # Rotate
                angle = float(input("Enter the angle of rotation: "))
                result = rotate_image(image, angle)
                cv2.imshow("Rotated Image", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            elif img_choice == '2':  # Scale
                scale_factor = float(input("Enter the scale factor: "))
                result = scale_image(image, scale_factor)
                cv2.imshow("Scaled Image", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            elif img_choice == '3':  # Reflect
                axis = input("Enter the axis of reflection (x or y): ").lower()
                result = reflect_image(image, axis)
                cv2.imshow("Reflected Image", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            elif img_choice == '4':  # Shear
                axis = input("Enter the axis of shear (x or y): ").lower()
                shear_factor = float(input(f"Enter the shear factor along {axis.upper()} axis: "))
                result = shear_image(image, shear_factor, axis)
                cv2.imshow("Sheared Image", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            elif img_choice == '5':  # Custom Transformation
                matrix = input("Enter the transformation matrix as comma-separated values (e.g., 1,0,0,1,0,0 for 2x3 matrix): ")
                matrix = np.array(matrix.split(','), dtype=float).reshape((2, 3))
                result = transform_image(image, matrix)
                cv2.imshow("Transformed Image", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Invalid choice. Please try again.")
        
        else:
            print("Invalid choice. Please try again.")

menu()