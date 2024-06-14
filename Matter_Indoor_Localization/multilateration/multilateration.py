import numpy as np
from scipy.optimize import least_squares

def multilateration(anchors, distances):
    """
    Perform multilateration to determine the 3D position of a tracker.
    
    Parameters:
    anchors (np.ndarray): A numpy array of shape (N, 3) where N is the number of anchors,
                          and each row represents the (x, y, z) coordinates of an anchor.
    distances (np.ndarray): A numpy array of shape (N,) where each element represents the
                            distance from the tracker to the corresponding anchor.
    
    Returns:
    np.ndarray: The estimated (x, y, z) coordinates of the tracker.
    """
    
    # Initial guess for the tracker position
    initial_position = np.mean(anchors, axis=0)
    
    # Define the function that calculates the residuals
    def residuals(tracker_pos):
        return np.linalg.norm(anchors - tracker_pos, axis=1) - distances
    
    # Perform least squares optimization
    result = least_squares(residuals, initial_position)
    
    # Return the estimated position
    return result.x

# Example usage
anchors = np.array([
    [0.0, 0.0, 0.0],
    [160.0, 0.0, 0.0],
    [160.0, 80.0, 0.0],
    [0.0, 80.0, 0.0]
])

distances = np.array([1.0, 1.414, 1.414, 1.732])

tracker_position = multilateration(anchors, distances)
print("Estimated tracker position:", tracker_position)