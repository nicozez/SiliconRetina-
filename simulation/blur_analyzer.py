import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

original = cv2.imread(os.path.join(script_dir, 'display_og_00000.BMP'))
if original is None:
    raise FileNotFoundError("Could not load display_og_00000.BMP")

blurred_images = []
for i in [1, 2, 3, 4, 5]:
    img = cv2.imread(os.path.join(script_dir, f'display_{i}_00000.BMP'))
    if img is None:
        raise FileNotFoundError(f"Could not load display_{i}_00000.BMP")
    blurred_images.append(img)

iteration_counts = [1, 2, 3, 4, 5]

estimated_sigmas = []
for i, blurred in enumerate(blurred_images):
    best_sigma = None
    best_score = float('inf')
    sigma_range = np.linspace(0.1, 20, 1000)
    
    # Add progress bar for sigma estimation
    for sigma in tqdm(sigma_range, desc=f'Processing iteration {i+1}/{len(blurred_images)}'):
        simulated = gaussian_filter(original, sigma=sigma)
        error = np.mean((simulated - blurred) ** 2)
        if error < best_score:
            best_score = error
            best_sigma = sigma
    estimated_sigmas.append(best_sigma)

print(estimated_sigmas)
plt.plot(iteration_counts, np.array(estimated_sigmas)**2, 'o-')
plt.xlabel('Iterations')
plt.ylabel('Estimated σ²')
plt.title('σ² vs Iterations')
plt.grid(True)
plt.show()
