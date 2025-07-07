"""
oms_static.py

Run a single still image through the OFF-bipolar ➜ amacrine ➜ OMS-RGC
cascade of Baccus et al. (2008).

"""

import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#implement non-linear activation , what is the threshold and slope? 
def rectifier(v, thresh=0.0, slope=1e2):
    """Steep sigmoid ≈ hard threshold. Looks like a ReLU""" 
    return np.maximum(0.0, slope * (v - thresh))

def gaussian(size, sigma):
    ax = np.arange(-(size // 2), size // 2 + 1)
    g = np.exp(-0.5 * (ax / sigma) ** 2)
    return g / g.sum()
def difference_of_gaussians(size_c=1, sig_c=0.3, size_s=41, sig_s=15.0, w_surround=1.2):
    sz = max(size_c, size_s)
    pad_c = (sz - size_c) // 2
    pad_s = (sz - size_s) // 2
    center = np.pad(gaussian(size_c, sig_c), (pad_c, pad_c))
    surround = np.pad(gaussian(size_s, sig_s), (pad_s, pad_s))
    return center - w_surround * surround

# setup params for OMS; spatial filter, amacrine gain, rectification threshold, rectification slope, RGC threshold
class OMSStatic:
    """
    1. OFF bipolars:     linear DoG RF  (space only)
    2. Rectification:    synaptic non-linearity per bipolar terminal
    3. Amacrine:         sums rectified background, divides object drive
    4. OMS-RGC:          optional spike threshold
    """

    def __init__(self, beta=1.04, rect_thresh=0.0, rect_slope=1e2, rgc_thresh = 0.01):
        self.spatial = difference_of_gaussians()
        self.beta = beta
        self.rect_thresh = rect_thresh
        self.rect_slope = rect_slope
        self.rgc_thresh = rgc_thresh

    # ---------- public ----------------------------------------------------
    def process(self, frame, obj_mask, bkg_mask):
        
        """
        frame     : 2-D grayscale ndarray (H,W) in 0-1 range
        obj_mask  : boolean (H,W) region supplying excitatory drive
        bkg_mask  : boolean (H,W) region supplying amacrine inhibition
        returns   : dict { 'E', 'A', 'V', 'spike', 'lin', 'rect' }
        """

        # Stage 1: spatial convolution - output of OFF bipolar cells ; np.outer creates a 2d filter from 1d kernel
        lin = signal.fftconvolve(frame, np.outer(self.spatial, self.spatial), mode='same')

        # Stage 2: rectification per pixel; turns into DC signal, non-linearity , every bipolar must be rectified 
        rect = rectifier(lin, self.rect_thresh, self.rect_slope)

        #sums the rectified signal within the object and background regions; 
        B = rect[obj_mask].sum()     # direct bipolar excitary input ; sum of spatiotemporal bipoalr cells 
        A = rect[bkg_mask].sum()     # background amacrine inhibtory drive 


        # Stage 3: gain control, fig. 5 , final membrane potential of the OMS cell
        # integ(s(x,y,t)*alphaF(x,y,t-tau)dxdydt) /  1+ Ba'(t)
        G = B / (1.0 + self.beta * A) 

        #print(f"G = {G:.10f}, rgc_thresh = {self.rgc_thresh:.10f}, G > rgc_thresh = {G > self.rgc_thresh}")

        #thresholding for RGC spike, -50mV usually 
        spike = int(G > self.rgc_thresh)
        return {'Excitatory Bipolar Signal': B, 'Inhibitory Amacrine Signal': A, 'RGC Output Voltage': G, 'spike': spike, 'lin': lin, 'rect': rect}

    def visualize(self, frame, lin, rect, obj_mask=None, G=None):
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        axes[0].imshow(frame, cmap='gray')
        axes[0].set_title('Input')
        axes[0].axis('off')
        axes[1].imshow(lin, cmap='gray')
        axes[1].set_title('Linear')
        axes[1].axis('off')
        axes[2].imshow(rect, cmap='gray')
        axes[2].set_title('Rectified')
        axes[2].axis('off')
        plt.tight_layout()
        plt.show()

#ignore except for otsu
    def detect_objects_adaptive(self, frame, method='threshold'):
        """
        Automatically detect objects in the image based on content
        method: 'threshold', 'edge', 'gradient', 'otsu'
        """
        if method == 'threshold':
            # Simple threshold-based segmentation
            thresh = np.mean(frame)
            obj_mask = frame < thresh  # Dark objects on light background
            bkg_mask = ~obj_mask
            
        #generally best; experimentally defines object vs background 
        elif method == 'otsu':
            # Otsu's method for automatic thresholding
            frame_uint8 = (frame * 255).astype(np.uint8)
            thresh, _ = cv2.threshold(frame_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            obj_mask = frame < (thresh / 255.0)
            bkg_mask = ~obj_mask
            
        elif method == 'edge':
            # Edge-based segmentation
            frame_uint8 = (frame * 255).astype(np.uint8)
            edges = cv2.Canny(frame_uint8, 50, 150)
            # Dilate edges to create regions
            kernel = np.ones((3,3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)
            obj_mask = edges > 0
            bkg_mask = ~obj_mask
            
        elif method == 'gradient':
            # Gradient-based segmentation
            grad_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            thresh = np.percentile(gradient_magnitude, 75)
            obj_mask = gradient_magnitude > thresh
            bkg_mask = ~obj_mask
            
        else:
            raise ValueError(f"Unknown detection method: {method}")
        
        return obj_mask, bkg_mask


# VISUALISATION 
if __name__ == "__main__":
    # Load image
    img = cv2.imread("black_squares.png", cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0 # divide everyhting by 255 to get into float

    # Initialize OMS system
    oms = OMSStatic(beta=1.0, rect_thresh=0.01, rgc_thresh=0.01)

    print("="*60)
    print("OMS ALGORITHM DEMO - Content-Based Object Detection")
    print("="*60)
    
    # Test different object detection methods
    detection_methods = [
        ('dark_regions', 'Dark regions as objects'),
        ('bright_regions', 'Bright regions as objects'),
        ('otsu', 'Otsu automatic thresholding'),
        ('threshold', 'Mean-based thresholding'),
        ('center_circle', 'Fixed center circle (original)')
    ]
    
    for method, description in detection_methods:
        print(f"\n{'-'*50}")
        print(f"Testing: {description}")
        print(f"{'-'*50}")
        
        try:
            if method in ['dark_regions', 'bright_regions', 'otsu', 'threshold']:
                obj_mask, bkg_mask = oms.detect_objects_adaptive(img, method)
            else:
                obj_mask, bkg_mask = oms.detect_objects_manual(img, method)
       
            result = oms.process(img, obj_mask, bkg_mask)
            oms.visualize(img, result['lin'], result['rect'], obj_mask=obj_mask, G=result['RGC Output Voltage'])
            
            # Print results
            print(f"Object pixels: {obj_mask.sum()}")
            print(f"Background pixels: {bkg_mask.sum()}")
            print(f"Excitatory drive: {result['Excitatory Bipolar Signal']:.3f}")
            print(f"Inhibitory drive: {result['Inhibitory Amacrine Signal']:.3f}")
            print(f"Final response: {result['RGC Output Voltage']:.3f}")
            print(f"Spike generated: {'YES' if result['spike'] else 'NO'}")
            
        except Exception as e:
            print(f"Error with method '{method}': {e}")
    
    print("DEMO COMPLETE")

