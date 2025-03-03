import cv2
import numpy as np
from pathlib import Path
import logging
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Reconstructor:
    def __init__(self, frames_dir='captured_frames'):
        self.frames_dir = Path(frames_dir)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.camera_params = []  # Store camera parameters for bundle adjustment
        self.points_3d = []      # Store 3D points for bundle adjustment
        self.point_colors = []   # Store colors for 3D points
        self.camera_indices = [] # Store camera indices for each observation
        self.point_indices = []  # Store point indices for each observation
        self.points_2d = []      # Store 2D observations
        
    def process_frames(self):
        """Process captured frames and extract features"""
        frame_paths = sorted(self.frames_dir.glob('*.jpg'))
        if not frame_paths:
            raise ValueError("No frames found in directory")

        self.logger.info(f"Found {len(frame_paths)} frames")
        sift = cv2.SIFT_create(nfeatures=2000)  # Increase number of features
        frames_data = []
        
        for frame_path in frame_paths:
            try:
                self.logger.info(f"Processing frame: {frame_path}")
                
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    self.logger.warning(f"Could not read frame: {frame_path}")
                    continue
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                keypoints, descriptors = sift.detectAndCompute(gray, None)
                self.logger.info(f"Found {len(keypoints)} keypoints in {frame_path}")
                
                frames_data.append({
                    'path': frame_path,
                    'frame': frame,
                    'keypoints': keypoints,
                    'descriptors': descriptors
                })
            except Exception as e:
                self.logger.warning(f"Error processing frame {frame_path}: {str(e)}")
                continue
        
        if not frames_data:
            raise ValueError("Could not process any frames. Check that your images are valid and in a supported format.")
            
        return frames_data

    def draw_matches(self, img1, kp1, img2, kp2, matches):
        """Draw matches between two images"""
        # Create a new output image that concatenates the two images
        rows1, cols1 = img1.shape[:2]
        rows2, cols2 = img2.shape[:2]
        
        # Resize images to have the same height
        height = min(rows1, rows2)
        ratio1 = height / rows1
        ratio2 = height / rows2
        
        img1_resized = cv2.resize(img1, (int(cols1 * ratio1), int(rows1 * ratio1)))
        img2_resized = cv2.resize(img2, (int(cols2 * ratio2), int(rows2 * ratio2)))
        
        out = np.zeros((height, img1_resized.shape[1] + img2_resized.shape[1], 3), dtype=np.uint8)
        out[:, :img1_resized.shape[1]] = img1_resized
        out[:, img1_resized.shape[1]:] = img2_resized
        
        # Draw lines between matches
        for match in matches:
            # Get the matching keypoints for each of the images
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx

            # x - columns, y - rows
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt
            
            # Scale points according to resized images
            x1 = int(x1 * ratio1)
            y1 = int(y1 * ratio1)
            x2 = int(x2 * ratio2)
            y2 = int(y2 * ratio2)
            
            # Draw a line with a circle at each end
            cv2.line(out, (x1, y1), (x2 + img1_resized.shape[1], y2), (0, 255, 0), 1)
            cv2.circle(out, (x1, y1), 3, (255, 0, 0), 1)
            cv2.circle(out, (x2 + img1_resized.shape[1], y2), 3, (255, 0, 0), 1)
            
        return out

    def match_features(self, frames_data):
        """Match features between consecutive frames and across all frames"""
        self.logger.info("Matching features between frames...")
        matches_data = []
        
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Match between consecutive frames and also with a sliding window
        window_size = 3  # Match with 2 previous frames
        
        self.logger.info(f"Found {len(frames_data)} valid frames")
        if len(frames_data) < 2:
            raise ValueError(f"Need at least 2 valid frames for reconstruction, but only found {len(frames_data)}")
        
        # Create window for visualization
        cv2.namedWindow('Feature Matches', cv2.WINDOW_NORMAL)
        
        for i in range(len(frames_data)):
            for j in range(max(0, i-window_size), i):
                desc1 = frames_data[j]['descriptors']
                desc2 = frames_data[i]['descriptors']
                
                if desc1 is None or desc2 is None:
                    continue
                    
                # knnMatch with k=2 for Lowe's ratio test
                matches = flann.knnMatch(desc1, desc2, k=2)
                
                good_matches = []
                try:
                    for m, n in matches:
                        if m.distance < 0.75 * n.distance:  # Slightly relaxed ratio for more matches
                            good_matches.append(m)
                except ValueError:
                    continue
                
                if len(good_matches) >= 8:
                    self.logger.info(f"Found {len(good_matches)} matches between frames {j} and {i}")
                    
                    # Visualize matches
                    img_matches = self.draw_matches(
                        frames_data[j]['frame'],
                        frames_data[j]['keypoints'],
                        frames_data[i]['frame'],
                        frames_data[i]['keypoints'],
                        good_matches
                    )
                    
                    # Show matches
                    cv2.imshow('Feature Matches', img_matches)
                    cv2.waitKey(1000)  # Display for 1 second
                    
                    matches_data.append({
                        'frame1_idx': j,
                        'frame2_idx': i,
                        'matches': good_matches
                    })
        
        cv2.destroyAllWindows()
        
        if not matches_data:
            raise ValueError("Could not find enough matches between frames. Try adjusting the matching parameters or capturing more overlapping images.")
            
        self.logger.info(f"Total matched frame pairs: {len(matches_data)}")
        return matches_data

    def estimate_camera_pose(self, kp1, kp2, matches, K):
        """Estimate camera pose between two frames"""
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return None, None, None, None, None
            
        points, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
        if points < 8:
            return None, None, None, None, None
            
        return R, t, mask, pts1[mask.ravel()==1], pts2[mask.ravel()==1]

    def triangulate_points(self, pts1, pts2, P1, P2):
        """Triangulate 3D points from matched features"""
        pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        
        # Handle division by zero and invalid values
        w = pts4D[3, :]
        mask = np.abs(w) > 1e-8
        pts3D = np.zeros((np.sum(mask), 3))  # Use sum(mask) to set correct size
        pts3D = (pts4D[:3, mask] / w[mask]).T
        
        return pts3D

    def filter_points(self, points_3d, max_reprojection_error=20):
        """Filter outlier points using statistical analysis"""
        if len(points_3d) < 4:
            return np.array([])
            
        # Reshape points for nearest neighbors
        points_reshaped = points_3d.reshape(-1, 3)
        
        # Remove points with NaN or Inf values
        valid_points = ~np.any(np.isnan(points_reshaped) | np.isinf(points_reshaped), axis=1)
        if not np.any(valid_points):
            return np.zeros(len(points_3d), dtype=bool)
            
        points_filtered = points_reshaped[valid_points]
        
        # Use nearest neighbors to identify outliers
        if len(points_filtered) < 2:
            return valid_points
            
        nbrs = NearestNeighbors(n_neighbors=min(10, len(points_filtered))).fit(points_filtered)
        distances, _ = nbrs.kneighbors(points_filtered)
        
        # Filter based on average distance to neighbors
        avg_distances = distances.mean(axis=1)
        threshold = np.mean(avg_distances) + 2 * np.std(avg_distances)
        neighbor_mask = avg_distances < threshold
        
        # Combine masks
        final_mask = np.zeros(len(points_3d), dtype=bool)
        final_mask[valid_points] = neighbor_mask
        
        return final_mask

    def project_points(self, points_3d, camera_params, K):
        """Project 3D points to 2D using camera parameters"""
        R = Rotation.from_rotvec(camera_params[:3]).as_matrix()
        t = camera_params[3:6]
        
        points_proj = np.dot(points_3d, R.T) + t
        points_proj = points_proj[:, :2] / points_proj[:, 2:]
        return np.dot(points_proj, K[:2, :2].T) + K[:2, 2]

    def objective_function(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
        """Objective function for bundle adjustment"""
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))
        
        projected = np.zeros((len(camera_indices), 2))
        for i in range(len(camera_indices)):
            projected[i] = self.project_points(points_3d[point_indices[i]].reshape(1, 3),
                                            camera_params[camera_indices[i]],
                                            K)
            
        return (projected - points_2d).ravel()

    def run_bundle_adjustment(self, K):
        """Run bundle adjustment to optimize camera parameters and 3D points"""
        self.logger.info("Running bundle adjustment...")
        
        n_cameras = len(set(self.camera_indices))
        n_points = len(set(self.point_indices))
        
        x0 = np.zeros(n_cameras * 6 + n_points * 3)
        
        # Initial camera parameters
        for i in range(n_cameras):
            x0[i * 6:(i + 1) * 6] = self.camera_params[i]
            
        # Initial 3D points
        x0[n_cameras * 6:] = self.points_3d.ravel()
        
        # Run optimization
        res = least_squares(self.objective_function, x0,
                          args=(n_cameras, n_points, np.array(self.camera_indices),
                                np.array(self.point_indices), np.array(self.points_2d), K),
                          verbose=2, x_scale='jac', ftol=1e-4, method='trf')
                          
        # Update parameters
        camera_params = res.x[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = res.x[n_cameras * 6:].reshape((n_points, 3))
        
        return camera_params, points_3d

    def create_point_cloud(self, frames_data, matches_data):
        """Create point cloud from matched features with bundle adjustment"""
        self.logger.info("Creating point cloud...")
        
        # Camera intrinsic parameters
        focal_length = max(frames_data[0]['frame'].shape[:2]) * 1.2  # Estimate focal length
        center = (frames_data[0]['frame'].shape[1] / 2, frames_data[0]['frame'].shape[0] / 2)
        K = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ])
        
        # Initialize first camera
        P1 = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0]])
        P1 = K.dot(P1)
        
        point_cloud = []
        point_colors = []
        camera_positions = []
        
        for match_data in matches_data:
            frame1 = frames_data[match_data['frame1_idx']]
            frame2 = frames_data[match_data['frame2_idx']]
            
            R, t, mask, pts1, pts2 = self.estimate_camera_pose(
                frame1['keypoints'],
                frame2['keypoints'],
                match_data['matches'],
                K
            )
            
            if R is None or pts1 is None or pts2 is None:
                continue
                
            # Store camera parameters for bundle adjustment
            rotvec = Rotation.from_matrix(R).as_rotvec()
            self.camera_params.append(np.concatenate([rotvec, t.ravel()]))
            camera_positions.append(-R.T.dot(t).ravel())
            
            # Create second camera matrix
            P2 = np.hstack((R, t))
            P2 = K.dot(P2)
            
            # Triangulate points
            points_3d = self.triangulate_points(pts1, pts2, P1, P2)
            
            # Filter points
            valid_mask = self.filter_points(points_3d)
            points_3d = points_3d[valid_mask]
            
            if len(points_3d) == 0:
                continue
            
            # Get colors and store correspondences
            for i, match in enumerate(match_data['matches']):
                if valid_mask[i]:
                    kp = frame1['keypoints'][match.queryIdx].pt
                    x, y = int(kp[0]), int(kp[1])
                    if 0 <= x < frame1['frame'].shape[1] and 0 <= y < frame1['frame'].shape[0]:
                        color = frame1['frame'][y, x]
                        point_colors.append(color)
                        point_cloud.append(points_3d[i])
                        
                        # Store correspondences for bundle adjustment
                        self.camera_indices.append(match_data['frame1_idx'])
                        self.point_indices.append(len(point_cloud) - 1)
                        self.points_2d.append(kp)
        
        if not point_cloud:
            raise ValueError("Could not reconstruct any 3D points. Try capturing more overlapping images.")
        
        point_cloud = np.array(point_cloud)
        point_colors = np.array(point_colors)
        camera_positions = np.array(camera_positions)
        
        if len(point_cloud) < 10:
            raise ValueError("Too few points reconstructed. Try capturing more overlapping images.")
        
        # Run bundle adjustment
        try:
            camera_params, points_3d = self.run_bundle_adjustment(K)
            # Update point cloud with optimized points
            point_cloud = points_3d
        except Exception as e:
            self.logger.warning(f"Bundle adjustment failed: {str(e)}. Using unoptimized points.")
        
        return point_cloud, point_colors, camera_positions

    def visualize_reconstruction(self, points_3d, colors, camera_positions):
        """Visualize the 3D reconstruction"""
        self.logger.info("Visualizing reconstruction...")
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points with actual colors
        colors_normalized = colors.astype(float) / 255
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                  c=colors_normalized, s=1)
        
        # Plot camera positions
        ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
                  c='red', marker='^', s=100, label='Camera Positions')
        
        # Add camera position numbers
        for i, pos in enumerate(camera_positions):
            ax.text(pos[0], pos[1], pos[2], str(i), color='red')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Reconstruction')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])
        
        # Add legend
        ax.legend()
        
        plt.show()

    def save_point_cloud(self, points, colors, output_file='point_cloud.ply'):
        """Save point cloud to PLY file"""
        self.logger.info(f"Saving point cloud to {output_file}...")
        
        with open(output_file, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for pt, color in zip(points, colors):
                f.write(f"{pt[0]} {pt[1]} {pt[2]} {int(color[2])} {int(color[1])} {int(color[0])}\n")

    def run_pipeline(self):
        """Run the complete reconstruction pipeline"""
        try:
            self.logger.info("Starting reconstruction pipeline")
            
            # Process frames and match features
            frames_data = self.process_frames()
            frames = [str(frame['path']) for frame in frames_data]
            frames = [str(frame.encode('utf-8'), errors='ignore') for frame in frames]
            matches_data = self.match_features(frames_data)
            
            if not matches_data:
                raise ValueError("Could not find enough matches between frames. Try capturing more overlapping images.")
            
            # Create point cloud with bundle adjustment
            points_3d, colors, camera_positions = self.create_point_cloud(frames_data, matches_data)
            
            # Visualize reconstruction
            self.visualize_reconstruction(points_3d, colors, camera_positions)
            
            # Save point cloud
            self.save_point_cloud(points_3d, colors)
            
            self.logger.info("Reconstruction complete! You can view the point cloud using MeshLab or CloudCompare")
            
        except Exception as e:
            self.logger.error(f"Error in reconstruction pipeline: {str(e)}")
            raise

if __name__ == "__main__":
    reconstructor = Reconstructor()
    reconstructor.run_pipeline()
