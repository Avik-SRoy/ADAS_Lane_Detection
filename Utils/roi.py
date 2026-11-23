import cv2
import numpy as np


def region_of_interest(image):
    """Apply a trapezoidal region-of-interest mask to a single-channel image."""
    height, width = image.shape[:2]

    polygon = np.array([[
        (int(width * 0.05), height),
        (int(width * 0.90), height),
        (int(width * 0.62), int(height * 0.65)),
        (int(width * 0.40), int(height * 0.65)),
    ]], np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(image, mask)


class PerspectiveTransformer:
    """Inverse Perspective Mapping (IPM) between camera view and Bird's-Eye View."""

    def __init__(self, width, height, src_points_norm, dst_points_norm):
        self.width = width
        self.height = height

        scale = np.array([width, height], dtype=np.float32)
        self.src_points = np.array(src_points_norm, dtype=np.float32) * scale
        self.dst_points = np.array(dst_points_norm, dtype=np.float32) * scale

        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

    def warp(self, image):
        """Warp camera-view image into top-down Bird's-Eye View."""
        return cv2.warpPerspective(
            image, self.M, (self.width, self.height), flags=cv2.INTER_LINEAR
        )

    def unwarp(self, image):
        """Project Bird's-Eye View image back into camera perspective."""
        return cv2.warpPerspective(
            image, self.Minv, (self.width, self.height), flags=cv2.INTER_LINEAR
        )
