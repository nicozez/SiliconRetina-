#!/usr/bin/env python3
import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_images(images, titles, cmap='gray'):
    n = len(images)
    plt.figure(figsize=(5 * n, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, n, i + 1)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.show()

def box_blur(img, ksize):
    """Averaging (box) filter."""
    return cv2.blur(img, (ksize, ksize))

def gaussian_blur(img, sigma):
    """Gaussian blur with given sigma."""
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def difference_of_gaussians(img, sigma_center, sigma_surround, weight_surround=1.0):
    """Difference of Gaussians (DoG) for center-surround."""
    center = gaussian_blur(img, sigma_center)
    surround = gaussian_blur(img, sigma_surround)
    return center - weight_surround * surround

def retina_model(img, 
                 box1=3, box2=7, 
                 sigma_center=1, sigma_surround=5, 
                 plot=False):
    # 1. Photoreceptor: normalize
    photoreceptor = img.astype(np.float32) / 255.0

    # 2. Horizontal cells: local average
    horizontal = box_blur(photoreceptor, box1)

    # 3. Bipolar cells: centre-surround
    bipolar = photoreceptor - horizontal

    # 4. Amacrine cells: average bipolar output
    amacrine = box_blur(bipolar, box2)

    # 5. Ganglion cells: second centre-surround
    ganglion = bipolar - amacrine

    if plot:
        dog = difference_of_gaussians(photoreceptor, sigma_center, sigma_surround)
        show_images(
            [photoreceptor, horizontal, bipolar, amacrine, ganglion, dog],
            ['Input', 
             f'Horizontal ({box1}×{box1})', 
             'Bipolar', 
             f'Amacrine ({box2}×{box2})', 
             'Ganglion', 
             f'DoG (σ={sigma_center},{sigma_surround})']
        )

    # === FIXED: use np.ptp instead of ganglion.ptp() ===
    dynamic_range = np.ptp(ganglion)
    # avoid division by zero
    if dynamic_range == 0:
        dynamic_range = 1e-6

    # rescale ganglion to 0–255 for display
    gang_disp = np.clip((ganglion - ganglion.min()) / dynamic_range, 0, 1)
    gang_disp = (gang_disp * 255).astype(np.uint8)
    return gang_disp

if __name__ == "__main__":
    # --- Webcam capture setup ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        exit(1)

    # Model parameters (tweak as desired)
    box1           = 7 #was 5 horizontal-cell kernel size
    box2           = 11  # amacrine-cell kernel size
    sigma_center   = 3   # DoG center sigma
    sigma_surround = 5   # DoG surround sigma

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # run the retina model
        ganglion_out = retina_model(
            gray, 
            box1=box1, 
            box2=box2, 
            sigma_center=sigma_center, 
            sigma_surround=sigma_surround,
            plot=False
        )

        # display
        cv2.imshow('Webcam Input', gray)
        cv2.imshow('Retina Ganglion Output', ganglion_out)

        # quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
