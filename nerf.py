from calibrate import *
import utils as ut
from torch.utils.data import Dataset
import torch

# NOTE: UNCOMMENT LATER WHEN WE HAVE A MAIN THING SETUP!!!
# TEST_IMG_PATH = "data/test_images"
# TEST_IMG_OBJECT_PATH = "data/test_images_object"
# test_images = ut.load_imgs_from_repo(TEST_IMG_PATH)
# test_object_images = ut.load_imgs_from_repo(TEST_IMG_OBJECT_PATH)


SAMPLED_RAYS_PER_IMAGE = 100 # M

class RaysDataset(Dataset):

    def __init__(self, images, K, c2ws, device, train=True,):
        super().__init__()
        self.images = torch.tensor(images, dtype=torch.float32).to(device)
        self.K = K
        self.c2ws = c2ws
        self.train = train
        self.device = device

        self.n_images, self.H, self.W, _ = images.shape

        
    def sample_rays(self, N):
        """
        Samples N rays from images. To do this, we globally sample from flattened coordinate grid.
        """
        total_pixels = self.n_images * self.H * self.W
        random_indices = torch.randperm(total_pixels)
        sampled_indices = random_indices[:N]
        #sampled_indices = np.random.choice(total_pixels, size=N, replace=False)

        image_indices = sampled_indices // (self.H * self.W)
        pixels_indices = sampled_indices % (self.H * self.W)

        rows = pixels_indices // self.W  # v coordinate
        cols = pixels_indices % self.W   # u coordinate

        uv_coords = torch.column_stack([cols + 0.5, rows + 0.5])

        ray_origins = torch.zeros((N, 3), device=self.device)
        ray_directions = torch.zeros((N, 3), device=self.device)
        pixels = torch.zeros((N, 3), device=self.device)

        for i in torch.unique(image_indices):
            mask = image_indices == i
            indices_for_i = torch.where(mask)[0]

            if len(indices_for_i) > 0:
                uv_batch = uv_coords[indices_for_i]
                r_o, r_d = pixel_to_ray(self.K, self.c2ws[i], uv_batch)
                r_o = torch.tensor(r_o, dtype=torch.float32, device=self.device)
                r_d = torch.tensor(r_d, dtype=torch.float32, device=self.device)
                
                ray_origins[indices_for_i] = r_o
                ray_directions[indices_for_i] = r_d

                batch_rows = rows[indices_for_i]
                batch_cols = cols[indices_for_i]
                # Correct indexing: images are [img, row, col, channel]
                pixels[indices_for_i] = self.images[i, batch_rows, batch_cols]
                
        return ray_origins, ray_directions, pixels


    def __getitem__(self, index):
        return self.images[index]
    
    def sample_all_rays_from_img(self, img_idx, c2w=None):
        all_pixel_indices = torch.arange(self.H * self.W)
        
        rows = all_pixel_indices // self.W  
        cols = all_pixel_indices % self.W  
        
        uv_coords = torch.column_stack([cols + 0.5, rows + 0.5])
        
        # if we provide a c2w, just use that transform directory
        if c2w is None:
            c2w = self.c2ws[img_idx]
        r_o, r_d = pixel_to_ray(self.K, c2w, uv_coords)
        r_o = torch.tensor(r_o, dtype=torch.float32, device=self.device)
        r_d = torch.tensor(r_d, dtype=torch.float32, device=self.device)

        
        return r_o, r_d
    
    def sample_partial_rays_from_img(self, n_rays, img_idx, c2w=None):
        all_pixel_indices = torch.arange(self.H * self.W)
        sampled_indices= torch.randperm(self.H * self.W)[:n_rays]
        all_pixel_indices = all_pixel_indices[sampled_indices]

        rows = all_pixel_indices // self.W  
        cols = all_pixel_indices % self.W  
        
        uv_coords = torch.column_stack([cols + 0.5, rows + 0.5])
        
        # if we provide a c2w, just use that transform directory
        if c2w is None:
            c2w = self.c2ws[img_idx]
        r_o, r_d = pixel_to_ray(self.K, c2w, uv_coords)
        r_o = torch.tensor(r_o, dtype=torch.float32, device=self.device)
        r_d = torch.tensor(r_d, dtype=torch.float32, device=self.device)

        pixels = self.images[img_idx, rows, cols]
        return r_o, r_d, pixels

    def __len__(self):
        return self.n_images



def parse_dataset(dataset_path):
    
    data = np.load(dataset_path)
    images_train = None
    c2ws_train = None
    images_val = None
    c2ws_val = None
    c2ws_test = None
    focal = None
    K = None

    # Training images: [100, 200, 200, 3]
    images_train = data["images_train"] / 255.0

    # Cameras for the training images 
    # (camera-to-world transformation matrix): [100, 4, 4]
    c2ws_train = data["c2ws_train"]

    # Validation images: 
    images_val = data["images_val"] / 255.0

    if "c2ws_val" in data:
        # Cameras for the validation images: [10, 4, 4]
        # (camera-to-world transformation matrix): [10, 200, 200, 3]
        c2ws_val = data["c2ws_val"]

    if "c2ws_test" in data:
        # Test cameras for novel-view video rendering: 
        # (camera-to-world transformation matrix): [60, 4, 4]
        c2ws_test = data["c2ws_test"]

    if "focal" in data:
        # Camera focal length
        focal = data["focal"]  # float

    if "K" in data:
        K = data["K"]
    return images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal, K

def get_K(focal_x, focal_y, o_x, o_y):
    K = np.eye(3)
    K[0][0] = focal_x
    K[1][1] = focal_y
    K[0][2] = o_x
    K[1][2] = o_y
    return K

def transform(c2w, x_c):
    """
    Transforms from camera to world coordinates using c2w
    """
    originally_singular = x_c.ndim == 1
    x_c = np.atleast_2d(x_c)

    N = x_c.shape[0]
    ones = np.ones((N, 1))
    x_c_homogeneous = np.hstack([x_c, ones])

    # To apply the transformation in batching, we apply c2w to every row of the x_c_homogenous matrix
    x_w_homogenous = (c2w @ x_c_homogeneous.T).T

    # Dehomogenize 
    x_w = x_w_homogenous[:, :3] / x_w_homogenous[:, 3:]

    return x_w[0] if originally_singular else x_w


def pixel_to_camera(K, uv, s):
    """
    Converts the pixel coordinates (u, v) into camera coordinates (x_c, y_c, z_c).

    K : (3, 3) Projection matrix (intrinsics)
    uv : (2, ) or (N, 2) array of pixel coordinates to unproject.
    s : (, ) or (N, ) array of corresponding depths 
    """
    originally_singular = uv.ndim == 1
    uv = np.atleast_2d(uv)

    N = uv.shape[0]

    ones = np.ones((N, 1))
    uv_hom = np.hstack([uv, ones])
    K_inv = np.linalg.inv(K)

    # Retrieve unscaled coordinates. 
    cam_coords_unscaled = (K_inv @ uv_hom.T).T

    # Scale coordinates by their depth
    cam_coords = cam_coords_unscaled * s
    return cam_coords[0] if originally_singular else cam_coords

def pixel_to_ray(K, c2w, uv):
    """
    Converts pixel coordinates uv (or batch of coordinates) into rays with origin and normalized direction
    """
    N = uv.shape[0]
    r_origin = c2w[:3, 3] # The last column is the translation vector. The translation vector is the origin of the camera for c2w
    r_origins = np.full((N, 3), c2w[:3, 3].flatten())
    world_coords = transform(c2w, pixel_to_camera(K, uv, s=1))
    norm_factor = np.linalg.norm(world_coords - r_origin, axis=1)
    r_d = (world_coords - r_origin) / norm_factor[:, np.newaxis]
    return r_origins, r_d


def sample_along_rays(rays_o, rays_d, perturb=True, near=2.0, far=6.0, n_samples=64,):
    # Sample points along this ray
    N = rays_o.shape[0]
    t_width = (far - near) / n_samples
    device = rays_o.device 
    t = near + torch.arange(n_samples, device=device) * t_width

    # Perturbs t to prevent overfitting when training
    if perturb:
        perturbation = torch.randn(N, n_samples, device=device) * t_width # offset by random perturbation. 
        t_perturbed = t[None, :] + perturbation
    else:
        t_perturbed = t[None, :].expand(N, n_samples)
    
    ray_points = rays_o[:, None, :] + (rays_d[:, None, :] * t_perturbed[..., None])
    all_ray_points = ray_points.reshape((-1, 3))
    return all_ray_points


def visualize_rays(images_train, c2ws_train, K, dataset=None, color=False):

    if dataset is None:
        dataset = RaysDataset(images_train, K, c2ws_train, device=None)
    rays_o, rays_d, pixels = dataset.sample_rays(100) # Should expect (B, 3)
    points = sample_along_rays(rays_o, rays_d, perturb=True, n_samples=32)

    rays_o = rays_o.cpu().numpy()
    rays_d = rays_d.cpu().numpy()
    points = points.cpu().numpy()

    if color:
        pixels_reshaped = np.repeat(np.expand_dims(pixels, axis=1), 32, axis=1)


    
    H, W = images_train.shape[1:3]
    print(H, W)

    print("K BEING USED FOR VISUALIZATION: ", K)
    # ---------------------------------------

    server = viser.ViserServer(share=True)
    #server.set_up_direction("-z")

    for i, (image, c2w) in enumerate(zip(images_train, c2ws_train)):
        server.add_camera_frustum(
            f"/cameras/{i}",
            fov=2 * np.arctan2(H / 2, K[0, 0]),
            aspect=W / H,
            scale=0.02,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=image
        )
    for i, (o, d) in enumerate(zip(rays_o, rays_d)):
        server.add_spline_catmull_rom(
            f"/rays/{i}", positions=np.stack((o, o + d * 6.0)),
        )
    if not color: 
        pixels_reshaped = np.zeros_like(points)
    server.add_point_cloud(
        f"/samples",
        colors=pixels_reshaped.reshape(-1, 3),
        points=points.reshape(-1, 3),
        point_size=0.02,
    )

    while True:
        time.sleep(0.1)  # Wait to allow visualization to run


def volrend(densities, rgbs, step_size):
    """
    Volumetrically renders pixels from predicted densities, color values, and sample size to sample along a ray.

    densities : torch tensor (N, sample_size, 1)
    rgbs : torch tensor (N, sample_size, 3)
    step_size : scalar denoting how we sample along a ray (a delta value)

    Returns : torch tensor (N, 3) of the volumetrically rendered colors from the N rays.
    """

    # Computing all the opacity values (alphas)
    alphas = 1 - torch.exp(-densities * step_size)

    # Computing the trasmittence values with torch.cumsum.
    # Note we shift and pad a 0 to model the "up to" affect on transmittence
    transmittences = torch.cumsum(densities * step_size, dim=1)
    cum_sum_shifted = transmittences[:, :-1, :]
    zero_tensor = torch.zeros_like(transmittences[:, :1, :])
    transmittences_final = torch.exp(-torch.cat([zero_tensor, cum_sum_shifted], dim=1))

    # The final rendered color is a weighted sum of predicted colors with their transmittence and opacity values as weights.
    rendered_color = torch.sum(transmittences_final * alphas * rgbs, dim=1)
    return rendered_color

def test_volrend():
    torch.manual_seed(42)
    sigmas = torch.rand((10, 64, 1))
    rgbs = torch.rand((10, 64, 3))
    step_size = (6.0 - 2.0) / 64
    rendered_colors = volrend(sigmas, rgbs, step_size)

    correct = torch.tensor([
        [0.5006, 0.3728, 0.4728],
        [0.4322, 0.3559, 0.4134],
        [0.4027, 0.4394, 0.4610],
        [0.4514, 0.3829, 0.4196],
        [0.4002, 0.4599, 0.4103],
        [0.4471, 0.4044, 0.4069],
        [0.4285, 0.4072, 0.3777],
        [0.4152, 0.4190, 0.4361],
        [0.4051, 0.3651, 0.3969],
        [0.3253, 0.3587, 0.4215]
    ])
    assert torch.allclose(rendered_colors, correct, rtol=1e-4, atol=1e-4)



def main():

    # Note this visualizes with a larger s. Adjust s in the viusalize_rays function to change this.

    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal, K = parse_dataset("data/nerf_data/lego_200x200.npz")
    H, W = images_train.shape[:2]
    K = get_K(focal, focal, W / 2, H / 2)
    visualize_rays(images_train, c2ws_train, K)

    dataset = RaysDataset(images_train, K, c2ws_train, train=True)
    rays_o, rays_d, pixels = dataset.sample_rays(100) # Should expect (B, 3)

    points = sample_along_rays(rays_o, rays_d, perturb=True)
    print(points.shape) 
    

if __name__ == "__main__":
    main()


