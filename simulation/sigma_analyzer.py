import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Given data
iterations = np.array([0, 1, 2, 3, 4, 5])
sigmas = np.array([0, 3.7055055055055055, 5.478378378378378, 6.7333333333333325, 7.789089089089089, 8.805005005005004])
sigmas = sigmas**2

# Model function
def sigma_model(n, a, b):
    return a * n + b

# Fit the curve
popt, pcov = curve_fit(sigma_model, iterations, sigmas)
a, b = popt

# Print results
print(f"a = {a:.6f}")
print(f"b = {b:.6f}")

# Optional: plot
n_vals = np.linspace(0, 5, 100)
sigma_vals = sigma_model(n_vals, a, b)

plt.plot(iterations, sigmas, 'o', label='Estimated sigmas')
plt.plot(n_vals, sigma_vals, '-', label=f'Fit: σ² = {a:.3f}n + {b:.3f}')
plt.xlabel('Iterations')
plt.ylabel('σ²')
plt.title('σ² vs. Diffusion Iterations')
plt.legend()
plt.grid(True)
plt.show()
