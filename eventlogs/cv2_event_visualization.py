#!/usr/bin/env python3
"""
Interactive 3D Event Data Visualization

This script loads event data from HDF5 files and provides an interactive interface
to select time windows and visualize events in 3D space (X, Y, Time, Polarity).
Uses efficient HDF5 data access for fast loading.
"""

import argparse
import h5py
import numpy as np
import cv2
from event_loader import EventDataLoader

def colorize_events(events: np.ndarray) -> np.ndarray:
    # Create a 3-channel color image (BGR format for OpenCV)
    h, w = events.shape
    events_display = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Gray (0.5) → White (255, 255, 255)
    gray_mask = (events == 0)
    events_display[gray_mask] = [255, 255, 255]
    
    # White (1.0) → Blue (0, 0, 255) in BGR format
    white_mask = (events == 1)
    events_display[white_mask] = [255, 0, 0]
    
    # Black (0.0) → Red (255, 0, 0) in BGR format
    black_mask = (events == -1)
    events_display[black_mask] = [0, 0, 255]
    
    return events_display

if __name__ == "__main__":
    """Main function to handle command line arguments and run visualization."""
    
    parser = argparse.ArgumentParser(description='Interactive 3D Event Data Visualization')
    parser.add_argument('h5_file', help='Path to HDF5 file containing event data')
    parser.add_argument('--start-time', '-t', type=float, default=None,
                       help='Start time in microseconds (for static mode)')
    parser.add_argument('--duration', '-d', type=float, default=None,
                       help='Duration in microseconds (for static mode)')
    
    args = parser.parse_args()
    

    with h5py.File(args.h5_file, 'r') as f:
        loader = EventDataLoader(f)
        loader.print_metadata()
        height = f['images'].shape[1]
        width = f['images'].shape[2]
        
        start_time = loader.get_start_time()
        end_time = loader.get_end_time()
        current_time = start_time

        while current_time < end_time:
            frame = loader.aggregate_time_range(current_time, current_time + args.duration)
            events_display = colorize_events(frame)
            cv2.imshow('2D Event Aggregate', events_display)
            current_time += args.duration

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break



