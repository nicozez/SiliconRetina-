import cv2
import numpy as np
from scipy import signal

def gaussian(size, sigma):
    ax = np.arange(-(size // 2), size // 2 + 1)
    g = np.exp(-0.5 * (ax / sigma) ** 2)
    return g / g.sum()
#size_c=1, sig_c=0.3, size_s=41, sig_s=15.0, w_surround=1.2)
def difference_of_gaussians(size_c=1, sig_c=0.3, size_s=41, sig_s=15.0, w_surround=1.2):
    sz = max(size_c, size_s)
    pad_c = (sz - size_c) // 2
    pad_s = (sz - size_s) // 2
    center = np.pad(gaussian(size_c, sig_c), (pad_c, pad_c))
    surround = np.pad(gaussian(size_s, sig_s), (pad_s, pad_s))
    return center - w_surround * surround

def rectifier(v, thresh=0.0, slope=1e2):
    return np.maximum(0.0, slope * (v - thresh))

def crop_center(img, crop_percent=0.8):
    h, w = img.shape[:2]
    ch, cw = int(h * crop_percent), int(w * crop_percent)
    start_y = (h - ch) // 2
    start_x = (w - cw) // 2
    return img[start_y:start_y+ch, start_x:start_x+cw]

#brownian motion
def random_walk_step(pos, bounds, stride=4):
    dy, dx = np.random.randint(-stride, stride+1, size=2)
    new_y = np.clip(pos[0] + dy, 0, bounds[0])
    new_x = np.clip(pos[1] + dx, 0, bounds[1])
    return new_y, new_x

SAVE_OUTPUTS = False

#MAIN -----
video_path = "Movie_Data/Sample.mp4"  
cap = cv2.VideoCapture(video_path)

crop_percent = 0.9
stride = 3
ret, frame = cap.read()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame_h, frame_w = gray.shape[:2]
ch, cw = int(frame_h * crop_percent), int(frame_w * crop_percent)
max_y, max_x = frame_h - ch, frame_w - cw
pos = [max_y // 2, max_x // 2]  # Start in the middle

dog_kernel_1d = difference_of_gaussians(size_c=1, sig_c=0.3, size_s=41, sig_s=15.0, w_surround=1.2)
#dog_kernel_1d = difference_of_gaussians(size_c=4, sig_c=0.5, size_s=9, sig_s=2.0, w_surround=1.0)
dog_kernel = np.outer(dog_kernel_1d, dog_kernel_1d)
prev_rect = None

# Prepare video writer for event output
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # fallback
if SAVE_OUTPUTS:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    event_writer = cv2.VideoWriter('event_output.mp4', fourcc, fps, (frame_w, frame_h), isColor=False)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind to first frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_norm = gray.astype(np.float32) / 255.0

    #cropped and shaken 
    cropped = crop_center(gray_norm, crop_percent)
    pos = random_walk_step(pos, (max_y, max_x), stride=stride)
    jittered = np.zeros_like(gray_norm)
    jittered[pos[0]:pos[0]+ch, pos[1]:pos[1]+cw] = cropped
    jittered_disp = (jittered * 255).astype(np.uint8)

    # (DoG), spatiotemporal filter 
    lin = signal.fftconvolve(jittered, dog_kernel, mode='same')
    lin_disp = lin - lin.min()
    #normalize 
    if lin_disp.max() > 0:
        lin_disp /= lin_disp.max()
    lin_disp = (lin_disp * 255).astype(np.uint8)

    # Rectified, keep the original lin value for computation 
    rect = rectifier(lin, thresh=0.00, slope=1e2)
    rect_disp = rect - rect.min()
    if rect_disp.max() > 0:
        rect_disp /= rect_disp.max()
    rect_disp = (rect_disp * 255).astype(np.uint8)
    
    H, W = rect.shape
    # init
    if prev_rect is None:
        prev_rect = np.zeros_like(rect)
    
    deltaV = rect - prev_rect
    event_threshold = 0.5
    # Adaptive thresholding idea 
    potential_events = deltaV > event_threshold
    
    event_img = np.zeros_like(rect, dtype=np.uint8)
    
    #only process pixels that are events, cross the threshold
    if np.any(potential_events):
        # Get coords
        event_coords = np.where(potential_events)
        neighbor_averages = np.zeros_like(deltaV)

        # Create a kernel that averages neighbors (excluding center)
        kernel_size = 21  # 10pixs on each side,
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        center_idx = kernel_size // 2
        kernel[center_idx, center_idx] = 0  # Exclude center 
        kernel = kernel / (kernel_size * kernel_size - 1)  # Normalize
        
        # Apply conv to get averages 
        neighbor_averages = signal.fftconvolve(deltaV, kernel, mode='same')
        
        local_threshold = 1000.0
        min_abs_delta = 0.05
        local_events = (deltaV > neighbor_averages * local_threshold) & (deltaV > min_abs_delta) & potential_events

        event_img[local_events] = 255
    
    # Erode to remove small noise, morphological noise
    #kernel = np.ones((3, 3), np.uint8)
    #event_img = cv2.erode(event_img, kernel, iterations=1)
    
    prev_rect = rect.copy()
    
    
    if SAVE_OUTPUTS:
        event_writer.write(event_img)
    
    # Debug print
    frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
    num_potential = np.sum(potential_events)
    num_local = np.sum(event_img > 0)
    print(f"Frame {frame_num}: Potential events: {num_potential}, Local events: {num_local}")
    
    # Frame differencing on rectified (COMMENTED OUT)
    #if prev_rect is not None:
    ##    diff = cv2.absdiff(rect, prev_rect)
    ##    thresh_val = np.percentile(diff, 99.9)
    #    event_img = np.zeros_like(diff, dtype=np.uint8)
    #    event_img[diff > thresh_val] = 255
    #    cv2.imshow('Event Output (Rectified Frame Diff)', event_img)
    #    if SAVE_OUTPUTS:
    ##        event_writer.write(event_img)
    #prev_rect = rect.copy()

    cv2.imshow('Jittered Frame', jittered_disp)
    cv2.imshow('Lin', lin_disp)
    cv2.imshow('Rect', rect_disp)
    cv2.imshow('Event Output (Per-Pixel Local)', event_img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if SAVE_OUTPUTS:
    event_writer.release()
cv2.destroyAllWindows() 