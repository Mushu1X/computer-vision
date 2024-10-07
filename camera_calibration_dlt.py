import numpy as np

def normalize_points(points):
    """
    Normalize 2D or 3D points for numerical stability in DLT.
    """
    n = points.shape[0]
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    if points.shape[1] == 3:
        scale = np.sqrt(3) / np.mean(np.linalg.norm(centered, axis=1))
        T = np.array([
            [scale, 0,     0,     -scale * centroid[0]],
            [0,     scale, 0,     -scale * centroid[1]],
            [0,     0,     scale, -scale * centroid[2]],
            [0,     0,     0,     1]
        ])
    else:
        scale = np.sqrt(2) / np.mean(np.linalg.norm(centered, axis=1))
        T = np.array([
            [scale, 0,     -scale * centroid[0]],
            [0,     scale, -scale * centroid[1]],
            [0,     0,     1]
        ])
    normalized_points = (T @ np.column_stack((points, np.ones(n))).T).T
    return normalized_points[:, :-1], T

def calibrate_camera(object_points, image_points):
    """
    Calibrate camera using the Direct Linear Transformation method.
    :param object_points: Nx3 array of 3D object points.
    :param image_points: Nx2 array of corresponding 2D image points.
    :return: Camera projection matrix P (3x4).
    """
    # Normalize the points
    obj_pts_norm, T_obj = normalize_points(object_points)
    img_pts_norm, T_img = normalize_points(image_points)

    n = object_points.shape[0]
    A = []

    for i in range(n):
        X, Y, Z = obj_pts_norm[i]
        x, y = img_pts_norm[i]
        A.append([
            -X, -Y, -Z, -1,  0,  0,  0,  0, x * X, x * Y, x * Z, x
        ])
        A.append([
             0,  0,  0,  0, -X, -Y, -Z, -1, y * X, y * Y, y * Z, y
        ])

    A = np.array(A)
    # Solve the equation A * p = 0 using SVD
    U, S, Vh = np.linalg.svd(A)
    P_normalized = Vh[-1].reshape(3, 4)

    # Denormalize the projection matrix
    P = np.linalg.inv(T_img) @ P_normalized @ T_obj

    # Normalize so that the last element is 1
    P /= P[-1, -1]

    return P

# Example usage
if __name__ == "__main__":
    # Sample 3D object points
    object_points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 2, 0],
        [0, 2, 0],
        [0, 0, -1],
        [1, 0, -1],
        [1, 2, -1],
        [0, 2, -1]
    ])

    # Corresponding 2D image points
    image_points = np.array([
        [472, 105],
        [508, 109],
        [515, 161],
        [476, 157],
        [470, 102],
        [506, 105],
        [513, 158],
        [474, 154]
    ])

    # Perform camera calibration
    P = calibrate_camera(object_points, image_points)
    print("Camera Projection Matrix P:\n", P)
