import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    eigenvector = np.random.rand(data.shape[0])
    for _ in range(num_steps):
        product = np.dot(data, eigenvector)
        eigenvector = product / np.linalg.norm(product)
    eigenvalue = float(np.dot(np.dot(eigenvector, data), eigenvector))

    return eigenvalue, eigenvector