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

distances_data = [
    [174, 52, 22, 171],
    [174, 45, 19, 175],
    [175, 48, 18, 170],
    [175, 41, 23, 174],
    [175, 41, 16, 170],
    [178, 44, 16, 173],
    [175, 44, 22, 177],
    [173, 43, 18, 174],
    [174, 48, 15, 172],
    [174, 45, 20, 171],
    [173, 46, 20, 177],
    [175, 48, 51, 169],
    [174, 49, 41, 174],
    [179, 47, 42, 170],
    [173, 45, 20, 169],
    [174, 44, 20, 171],
    [174, 45, 18, 171],
    [174, 46, 15, 171],
    [173, 42, 22, 172],
    [53,  46, 21, 170],
    [174, 42, 39, 172],
    [174, 47, 29, 169],
    [172, 43, 24, 169],
    [174, 46, 30, 164],
    [173, 43, 39, 167],
    [174, 47, 25, 164],
    [173, 45, 23, 165],
    [174, 46, 36, 168],
    [173, 48, 22, 167],
    [174, 45, 24, 172],
    [175, 42, 24, 171],
    [173, 48, 27, 172],
    [175, 44, 30, 170],
    [178, 44, 19, 174],
    [174, 44, 26, 173],
    ]

distances = np.array(distances_data)
tracker_positions = []
for sample in distances:
    tracker_position = multilateration(anchors, sample)
    tracker_positions.append(tracker_position)

mean_position = np.mean(tracker_positions, axis=0)
print("Estimated tracker position:", mean_position)