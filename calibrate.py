import cv2
import numpy as np
import utils as ut
import viser
import time
from sklearn.model_selection import train_test_split


TEST_IMG_PATH = "data/test_images"
TEST_IMG_OBJECT_PATH = "data/test_images_object"

AR_TAG_PATH = "data/phone_images/tags"
OBJECT_TAG_PATH = "data/phone_images/object"
ARUCO_TAG = cv2.aruco.DICT_4X4_50
TAG_SIZE = 0.06

# Create ArUco dictionary and detector parameters (4x4 tags)
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_TAG)
aruco_params = cv2.aruco.DetectorParameters()

object_corner_set = np.array([
    [0.0, TAG_SIZE, 0.0],      # Top-left (Y is "up" in image)
    [TAG_SIZE, TAG_SIZE, 0.0], # Top-right
    [TAG_SIZE, 0.0, 0.0],      # Bottom-right
    [0.0, 0.0, 0.0]            # Bottom-left
], dtype=np.float32)

def load_images(images_path, object_images_path):
    images = ut.load_imgs_from_repo(images_path)
    object_images = ut.load_imgs_from_repo(object_images_path)
    return images, object_images


def get_aruco_corners(img):
    # NOTE: Returns in TL, TR, BR, BL ORDER
    corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
    return corners, ids


def calibrate_camera(images):

    all_image_corners = []
    all_object_corners = []
    gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    


    for idx, img in enumerate(images):
        image_corners = []
        object_corners = []

        corners, ids = get_aruco_corners(img)

        # Check if any markers were detected
        if ids is not None:
            for corner_idx, corner_set in enumerate(corners):
                image_corners.append(corner_set[0])
                object_corners.append(object_corner_set)
            
            image_corners = np.array(image_corners, dtype=np.float32)
            object_corners = np.array(object_corners, dtype=np.float32)
            image_corners = np.reshape(image_corners, (-1, *image_corners.shape[2:]))
            object_corners = np.reshape(object_corners, (-1, *object_corners.shape[2:]))


            all_image_corners.append(image_corners)
            all_object_corners.append(object_corners)

    print("Calibrating Camera...")
    ret, camera_mat, distortion, rotation_vectors, translation_vectors = cv2.calibrateCamera(all_object_corners, all_image_corners, gray.shape[::-1], None, None)
    print("Calibration Complete!")
    return camera_mat, distortion


def get_extrinsics(img, K, distortion_coeff):
    """
    Solves for the Extrinsics of the camera taking the given image using Perspective-n-Point (PnP).
    Specifically returns the Camera->World transformation.
    """

    corners, ids = get_aruco_corners(img)
    if ids is None:
        print("No markers detected in get_extrinsics, skipping image.")
        return None
    


    # Solves for World->Camera
    success, rvec, tvec = cv2.solvePnP(object_corner_set, corners[0][0], K, distortion_coeff)
    if not success:
        print("Extrinsics Could Not be Found with PnP")
    else:
        T = np.eye(4)
        rot_mat = cv2.Rodrigues(rvec)[0]
        T[:3, :3] = rot_mat
        T[:3, 3] = tvec.flatten()

        # Retrun Camera->World
        return np.linalg.inv(T)
        


def visualize_pose_estimation(images, K, distortion_coeff):

    server = viser.ViserServer(share=True)
    server.scene.set_up_direction("-z")


    for i, img in enumerate(images):

        c2w = get_extrinsics(img, K, distortion_coeff)
        if c2w is None:
            continue
        
        H, W = img.shape[:2]

        # Example of visualizing a camera frustum (in practice loop over all images)
        # c2w is the camera-to-world transformation matrix (3x4), and K is the camera intrinsic matrix (3x3)
        server.scene.add_camera_frustum(
            f"/cameras/{i}", # give it a name
            fov=2 * np.arctan2(H / 2, K[0, 0]), # field of view
            aspect=W / H, # aspect ratio
            scale=0.01, # scale of the camera frustum change if too small/big
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz, # orientation in quaternion format
            position=c2w[:3, 3], # position of the camera
            image=img # image to visualize
        )

    while True:
        time.sleep(0.1)  # Wait to allow visualization to run

def build_dataset(calibration_images, object_images, out_path, target_size = (200, 200), undistort=False):

    K, distortion_coeff = calibrate_camera(calibration_images)
    H_old, W_old = calibration_images[0].shape[:2]
    focal_old = K[0, 0]
    N = len(calibration_images)

    if undistort:
        K_optimal, roi = cv2.getOptimalNewCameraMatrix(
            K,
            distortion_coeff, 
            (W_old, H_old),
            alpha=0,
            newImgSize=(W_old, H_old)
        )
        roi_x, roi_y, roi_w, roi_h = roi

    # If we provide target size, we downsample and that means we must adjust intrinsics accordingly
    if target_size is not None:
        H_new, W_new = target_size
        scaling_x = W_new / W_old
        scaling_y = H_new / H_old

        # adjust our intrinsics
        K_new = K.copy()
        K_new[0, 0] *= scaling_x
        K_new[1, 1] *= scaling_y
        K_new[0, 2] *= scaling_x
        K_new[1, 2] *= scaling_y

        focal_new = K_new[0, 0]
        print(f"Downsampling to {target_size}. Old focal: {focal_old}, New focal: {focal_new}")

        K = K_new

    else:
        print(f"Using original image resolution. Focal: {focal_old}")
        target_size = None 
        focal_new = focal_old

    all_images = []
    all_transforms = [] # Holds all the c2w transforms
    
    for img in object_images:
        if undistort:
            undistorted_img = cv2.undistort(img, K, distortion_coeff)

            undistorted_img = undistorted_img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        else:
            undistorted_img = img

        # Downsample
        if target_size is not None:
            undistorted_img_final = cv2.resize(
                img, 
                target_size,
                interpolation=cv2.INTER_AREA
            )
        else:
            undistorted_img_final = undistorted_img

        # compute extrinsics with the new K, new image
        c2w = get_extrinsics(undistorted_img_final, K, distortion_coeff)
        

        if c2w is not None:
            all_transforms.append(c2w)
            all_images.append(undistorted_img_final)


    all_images = np.array(all_images)
    all_transforms = np.array(all_transforms)

    train_images, val_images, train_transforms, val_transforms = train_test_split(
        all_images, 
        all_transforms, 
        test_size=0.1, # 10% validation split,
        shuffle=True    
    )  
    test_tranforms = np.zeros((N, 4, 4)) # NOTE: Filler data for testing right now

    print("K BEING SAVED TO DATASET: ", K)
    print("FOCAL BEING SAVED: ", focal_new)
    np.savez(
        out_path,
        images_train=train_images,    # (N_train, H, W, 3)
        c2ws_train=train_transforms,        # (N_train, 4, 4)
        images_val=val_images,        # (N_val, H, W, 3)
        c2ws_val=val_transforms,            # (N_val, 4, 4)
        c2ws_test=test_tranforms,          # (N_test, 4, 4)
        focal=focal_new,                  # float
        K = K
    )   
    print("Saved Dataset: ", out_path)
    return train_images, val_images, train_transforms, val_transforms



def main():
    
    # Swap the paths here to visualize different poses
    test_images, test_object_images = load_images(AR_TAG_PATH, OBJECT_TAG_PATH)

    K, distortion_coeff = calibrate_camera(test_images)
    get_extrinsics(test_images[0], K, distortion_coeff)
    visualize_pose_estimation(test_object_images, K, distortion_coeff)



if __name__ == "__main__":
    main()


