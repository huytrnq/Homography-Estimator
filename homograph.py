import numpy as np

class HomographyEstimator:
    def __init__(self):
        """
        Initialize the HomographyEstimator class.
        This class provides methods to compute homography matrices based on different transformation models and robust estimation using RANSAC.
        """
        self.model_methods = {
            'Euclidean': self._compute_euclidean,
            'Similarity': self._compute_similarity,
            'Affine': self._compute_affine,
            'Projective': self._compute_projective
        }

    def compute_homography(self, features, matches, model):
        """
        Compute the homography matrix based on the transformation model.

        Parameters:
            features (list): Points from source images.
            matches (list): Matched points in destination images.
            model (str): The transformation model ('Euclidean', 'Similarity', 'Affine', 'Projective').

        Returns:
            np.ndarray: The computed homography matrix.
        """
        if model in self.model_methods:
            return self.model_methods[model](features, matches)
        else:
            raise ValueError("Unsupported model type. Choose from 'Euclidean', 'Similarity', 'Affine', or 'Projective'.")

    def _compute_euclidean(self, features, matches):
        """
        Compute the Euclidean transformation homography matrix.

        Parameters:
            features (list): Source points.
            matches (list): Destination points.

        Returns:
            np.ndarray: The Euclidean homography matrix.
        """
        A = []
        b = []
        for (x, y), (x_d, y_d) in zip(features, matches):
            A.append([x, -y, 1, 0])
            A.append([y, x, 0, 1])
            b.extend([x_d, y_d])
        A = np.array(A)
        b = np.array(b)
        params = np.linalg.lstsq(A, b, rcond=None)[0]
        cos_theta, sin_theta, tx, ty = params
        R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        t = np.array([[tx], [ty]])
        H = np.vstack([np.hstack([R, t]), [0, 0, 1]])
        return H.astype(np.float32)

    def _compute_similarity(self, features, matches):
        """
        Compute the Similarity transformation homography matrix.

        Parameters:
            features (list): Source points.
            matches (list): Destination points.

        Returns:
            np.ndarray: The Similarity homography matrix.
        """
        A = []
        b = []
        for (x, y), (x_d, y_d) in zip(features, matches):
            A.append([x, -y, 1, 0])
            A.append([y, x, 0, 1])
            b.extend([x_d, y_d])
        A = np.array(A)
        b = np.array(b)
        params = np.linalg.lstsq(A, b, rcond=None)[0]
        s_cos_theta, s_sin_theta, tx, ty = params
        R = np.array([[s_cos_theta, -s_sin_theta], [s_sin_theta, s_cos_theta]])
        t = np.array([[tx], [ty]])
        H = np.vstack([np.hstack([R, t]), [0, 0, 1]])
        return H.astype(np.float32)

    def _compute_affine(self, features, matches):
        """
        Compute the Affine transformation homography matrix.

        Parameters:
            features (list): Source points.
            matches (list): Destination points.

        Returns:
            np.ndarray: The Affine homography matrix.
        """
        A = []
        b = []
        for (x, y), (x_d, y_d) in zip(features, matches):
            A.append([x, y, 0, 0, 1, 0])
            A.append([0, 0, x, y, 0, 1])
            b.extend([x_d, y_d])
        A = np.array(A)
        b = np.array(b)
        params = np.linalg.lstsq(A, b, rcond=None)[0]
        a, b, c, d, tx, ty = params
        H = np.array([[a, b, tx], [c, d, ty], [0, 0, 1]])
        return H.astype(np.float32)

    def _compute_projective(self, features, matches):
        """
        Compute the Projective transformation homography matrix.

        Parameters:
            features (list): Source points.
            matches (list): Destination points.

        Returns:
            np.ndarray: The Projective homography matrix.
        """
        A = []
        b = []
        for (x, y), (x_d, y_d) in zip(features, matches):
            A.append([x, y, 1, 0, 0, 0, -x*x_d, -y*x_d])
            A.append([0, 0, 0, x, y, 1, -x*y_d, -y*y_d])
            b.extend([x_d, y_d])
        A = np.array(A)
        b = np.array(b)
        params = np.linalg.lstsq(A, b, rcond=None)[0]
        a, b, c, d, e, f, g, h = params
        H = np.array([[a, b, c], [d, e, f], [g, h, 1]])
        return H.astype(np.float32)

    def ransac(self, features, matches, model, iterations=9, threshold=3.0):
        """
        RANSAC algorithm for robust homography estimation.

        Parameters:
            features (np.ndarray): Points from source images.
            matches (np.ndarray): Matched points in destination images.
            model (str): The transformation model ('Euclidean', 'Similarity', 'Affine', 'Projective').
            iterations (int): Number of iterations to run RANSAC.
            threshold (float): Distance threshold to consider a point as an inlier.

        Returns:
            np.ndarray: The best homography matrix found.
            list: List of inlier indices.
        """
        best_H = None
        max_inliers = 0
        best_inliers = []

        for _ in range(iterations):
            # Randomly select a subset of points (4 for projective, 3 otherwise)
            subset_size = 4 if model == 'Projective' else 3
            idx = np.random.choice(len(features), subset_size, replace=False)

            src_pts = features[idx]
            dst_pts = matches[idx]

            # Compute homography for the sampled points
            try:
                H = self.compute_homography(src_pts, dst_pts, model)
            except np.linalg.LinAlgError:
                continue

            # Calculate inliers
            current_inliers = []
            for i in range(len(features)):
                src_pt = np.array([features[i][0], features[i][1], 1.0])  # Homogeneous coordinates
                dst_estimated = np.dot(H, src_pt)
                dst_estimated /= dst_estimated[2]  # Normalize

                # Calculate distance
                distance = np.linalg.norm(dst_estimated[:2] - matches[i])
                if distance < threshold:
                    current_inliers.append(i)

            # Update the best homography if this iteration has more inliers
            if len(current_inliers) > max_inliers:
                max_inliers = len(current_inliers)
                best_H = H
                best_inliers = current_inliers

        return best_H, best_inliers

# Example Usage
# estimator = HomographyEstimator()
# H, inliers = estimator.ransac(features, matches, model='Projective')
