import numpy as np

def compute_homography(object_points, image_points):
    n = object_points.shape[0]
    A = []
    for i in range(n):
        X, Y = object_points[i]
        x, y = image_points[i]
        A.append([-X, -Y, -1,  0,  0,  0, x*X, x*Y, x])
        A.append([ 0,  0,  0, -X, -Y, -1, y*X, y*Y, y])
    A = np.array(A)
    _, _, Vh = np.linalg.svd(A)
    H = Vh[-1].reshape(3, 3)
    return H / H[-1, -1]

def compute_vij(H, i, j):
    return np.array([
        H[0,i]*H[0,j],
        H[0,i]*H[1,j] + H[1,i]*H[0,j],
        H[1,i]*H[1,j],
        H[2,i]*H[0,j] + H[0,i]*H[2,j],
        H[2,i]*H[1,j] + H[1,i]*H[2,j],
        H[2,i]*H[2,j]
    ])

def compute_intrinsics(H_list):
    V = []
    for H in H_list:
        V.append(compute_vij(H, 0, 1))
        V.append((compute_vij(H, 0, 0) - compute_vij(H, 1, 1)))
    V = np.array(V)
    _, _, Vh = np.linalg.svd(V)
    b = Vh[-1]
    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]]
    ])
    v0 = (B[0,1]*B[0,2] - B[0,0]*B[1,2]) / (B[0,0]*B[1,1] - B[0,1]**2)
    lambda_ = B[2,2] - (B[0,2]**2 + v0*(B[0,1]*B[0,2] - B[0,0]*B[1,2]))/B[0,0]
    alpha = np.sqrt(lambda_ / B[0,0])
    beta = np.sqrt(lambda_*B[0,0] / (B[0,0]*B[1,1] - B[0,1]**2))
    gamma = -B[0,1]*alpha**2*beta / lambda_
    u0 = gamma * v0 / beta - B[0,2]*alpha**2 / lambda_
    K = np.array([
        [alpha, gamma, u0],
        [0,     beta,  v0],
        [0,     0,     1]
    ])
    return K

def compute_extrinsics(K, H):
    K_inv = np.linalg.inv(K)
    lam = 1 / np.linalg.norm(np.dot(K_inv, H[:,0]))
    r1 = lam * np.dot(K_inv, H[:,0])
    r2 = lam * np.dot(K_inv, H[:,1])
    r3 = np.cross(r1, r2)
    t = lam * np.dot(K_inv, H[:,2])
    R = np.column_stack((r1, r2, r3))
    return R, t

if __name__ == "__main__":
    # Replace with your actual data
    object_points_all = [
        np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ]),
        # Add more arrays for each image
    ]
    image_points_all = [
        np.array([
            [320, 240],
            [400, 240],
            [400, 320],
            [320, 320]
        ]),
        # Add more arrays for each image
    ]

    H_list = []
    for obj_pts, img_pts in zip(object_points_all, image_points_all):
        H = compute_homography(obj_pts, img_pts)
        H_list.append(H)

    K = compute_intrinsics(H_list)
    extrinsics = []
    for H in H_list:
        R, t = compute_extrinsics(K, H)
        extrinsics.append((R, t))

    print("Intrinsic Matrix K:\n", K)
    for idx, (R, t) in enumerate(extrinsics):
        print(f"Extrinsics for Image {idx+1}:\nRotation:\n{R}\nTranslation:\n{t}")
