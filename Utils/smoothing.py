import numpy as np

class LaneSmoother:
    def __init__(self, alpha=0.8):
        self.alpha = alpha
        self.left_lane = None
        self.right_lane = None

    def smooth(self, new_left, new_right):

        if new_left is not None:
            new_left = np.array(new_left)  # 🔥 FIX
            if self.left_lane is None:
                self.left_lane = new_left
            else:
                self.left_lane = (
                    self.alpha * self.left_lane +
                    (1 - self.alpha) * new_left
                )

        if new_right is not None:
            new_right = np.array(new_right)  # 🔥 FIX
            if self.right_lane is None:
                self.right_lane = new_right
            else:
                self.right_lane = (
                    self.alpha * self.right_lane +
                    (1 - self.alpha) * new_right
                )

        return self.left_lane, self.right_lane