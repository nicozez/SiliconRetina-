#!/usr/bin/env python3
import cv2
import numpy as np
from matplotlib import pyplot as plt

def retina_model(img, prev_frame=None, threshold=0.05, receptive_field_size=3):
    # 1. Photoreceptors: Normalize current frame
    current = img.astype(np.float32) / 255.0
    
    # 2. Need previous frame for temporal comparison (otherwise return empty events and normalized frame)
    if prev_frame is None:
        return np.zeros_like(img), current
    
    # 3. Temporal difference
    temporal_diff = current - prev_frame
    
    # 4. Generate ON and OFF events based on temporal changes
    on_events = (temporal_diff > threshold).astype(np.float32)
    off_events = (temporal_diff < -threshold).astype(np.float32)
    
    # 6. Combine events
    events = np.full_like(temporal_diff, 0.5)  # Gray background (no events)
    events[on_events == 1] = 1.0   # White for ON events
    events[off_events == 1] = 0.0  # Black for OFF events
    
    # 7. Convert to display format
    events_display = (events * 255).astype(np.uint8)
    
    return events_display, current

# gray must become white, white must be red, black must be blue
def colorize_events(events: np.ndarray) -> np.ndarray:
    # Create a 3-channel color image (BGR format for OpenCV)
    h, w = events.shape
    events_display = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Gray (0.5) → White (255, 255, 255)
    gray_mask = (events == 127)
    events_display[gray_mask] = [255, 255, 255]
    
    # White (1.0) → Blue (0, 0, 255) in BGR format
    white_mask = (events == 255)
    events_display[white_mask] = [255, 0, 0]
    
    # Black (0.0) → Red (255, 0, 0) in BGR format
    black_mask = (events == 0.0)
    events_display[black_mask] = [0, 0, 255]
    
    return events_display

if __name__ == "__main__":
    # --- Webcam capture setup ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        exit(1)

    threshold = 0.05           # Minimum brightness change to trigger event
    receptive_field_size = 3   # Kernel size
    
    prev_frame = None
    
    print("'q' - quit")
    print("'+' - increase sensitivity (lower threshold)")
    print("'-' - decrease sensitivity (higher threshold)")
    print(f"Current threshold: {threshold:.3f}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        events_display, prev_frame = retina_model(
            gray,
            prev_frame=prev_frame,
            threshold=threshold,
            receptive_field_size=receptive_field_size
        )

        events_display_colorized = colorize_events(events_display)

        cv2.imshow('Webcam Input', gray)
        cv2.imshow('Events (Red=OFF, Blue=ON, White=No Change)', events_display_colorized)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+'):
            threshold = max(0.005, threshold - 0.005)  # Lower threshold = more sensitive
            print(f"Threshold: {threshold:.3f} (more sensitive)")
        elif key == ord('-'):
            threshold = min(0.2, threshold + 0.005)   # Higher threshold = less sensitive
            print(f"Threshold: {threshold:.3f} (less sensitive)")

    cap.release()
    cv2.destroyAllWindows()
