#!/usr/bin/env python3
"""
V2E Event Data Streaming Visualization

This script loads event data from V2E HDF5 files and provides streaming visualization
of events with adjustable time windows.
Uses efficient HDF5 data access for fast loading.
"""

import argparse
import h5py
import numpy as np
import sys
import cv2
from v2e_event_loader import V2EEventDataLoader

def colorize_events(events: np.ndarray) -> np.ndarray:
    """Colorize event frame for visualization.
    
    Args:
        events: 2D array with values in range [-1, 1]
        
    Returns:
        3-channel BGR image for OpenCV display
    """
    # Create a 3-channel color image (BGR format for OpenCV)
    h, w = events.shape
    events_display = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Neutral (0) → Gray (128, 128, 128)
    neutral_mask = (events == 0)
    events_display[neutral_mask] = [128, 128, 128]
    
    # Positive events (1.0) → Blue (255, 0, 0) in BGR format
    positive_mask = (events > 0)
    events_display[positive_mask] = [255, 0, 0]
    
    # Negative events (-1.0) → Red (0, 0, 255) in BGR format
    negative_mask = (events < 0)
    events_display[negative_mask] = [0, 0, 255]
    
    return events_display

def process_streaming_mode(loader: V2EEventDataLoader, args):
    """Process in streaming mode."""
    print("Streaming mode")
    print("Controls:")
    print("  - 'q': Quit")
    print("  - '+'/'-': Increase/decrease time window")
    print("  - Space: Pause/resume")
    
    start_time = loader.get_start_time_microseconds()
    end_time = loader.get_end_time_microseconds()
    current_time = start_time
    
    time_window = args.duration if args.duration else 50000  # 50ms default
    paused = False
    
    print(f"Time window: {time_window:.0f}μs")
    
    while current_time < end_time:
        if not paused:
            frame = loader.aggregate_time_range(current_time, current_time + time_window)
            events_display = colorize_events(frame)
            
            # Add timestamp info
            info_text = f"Time: {current_time:.0f}-{current_time+time_window:.0f}μs"
            info_text2 = f"({current_time:.3f}-{(current_time+time_window):.3f}s)"
            cv2.putText(events_display, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(events_display, info_text2, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow('V2E Event Stream', events_display)
            current_time += time_window
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # Pause/resume
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('=') or key == ord('+'):  # Increase time window
            time_window = min(time_window * 1.5, 1000000)  # Max 1 second
            print(f"Time window: {time_window:.0f}μs")
        elif key == ord('-'):  # Decrease time window
            time_window = max(time_window / 1.5, 1000)  # Min 1ms
            print(f"Time window: {time_window:.0f}μs")



if __name__ == "__main__":
    """Main function to handle command line arguments and run visualization."""
    
    parser = argparse.ArgumentParser(description='V2E Event Data Streaming Visualization')
    parser.add_argument('h5_file', help='Path to V2E HDF5 file containing event data')
    parser.add_argument('--width', '-w', type=int, default=None,
                       help='Image width (if not specified, read from file)')
    parser.add_argument('--height', '-H', type=int, default=None,
                       help='Image height (if not specified, read from file)')
    parser.add_argument('--duration', '-d', type=float, default=None,
                       help='Duration in microseconds for time window (default: 50000μs)')
    
    args = parser.parse_args()
    
    try:
        with h5py.File(args.h5_file, 'r') as f:
            # Set default dimensions if not specified
            width = args.width if args.width is not None else 640
            height = args.height if args.height is not None else 480
            
            loader = V2EEventDataLoader(f, height, width)
            loader.print_metadata()
            process_streaming_mode(loader, args)
                
    except FileNotFoundError:
        print(f"Error: File '{args.h5_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()
