#!/usr/bin/env python3
"""
Frame to Event Converter

This script watches a folder for newly created image frames, consumes them,
and appends events to a text eventlog file. Each event represents a brightness
level change with x, y, t coordinates and polarity (ON or OFF).

Usage:
    python f2e.py <base_path> <id> <eventlog.txt> <heartbeat_port>
"""

import argparse
import cv2
import numpy as np
import os
import re
import sys
import time
import glob

from f2e_utils import SupervisorClient, EventLogger, FileWatcher
from queue import Queue, Empty
from typing import Optional, Tuple

def find_latest_folder(base_dir: str, folder_id: str) -> Optional[str]:
    """Find the latest folder with the given ID."""
    pattern = os.path.join(base_dir, f"{folder_id}_*")
    matching_folders = glob.glob(pattern)

    if not matching_folders:
        return None

    directories = [f for f in matching_folders if os.path.isdir(f)]

    if not directories:
        return None

    directories.sort(key=lambda x: os.path.getctime(x), reverse=True)

    return directories[0]

def extract_events_from_image(image_path: str, time_delta: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    filename = os.path.basename(image_path)
    match = re.search(r'frame_(\d+)_(ON|OFF)', filename, re.IGNORECASE)
    if match is None:
        raise Exception()

    frame_number = int(match.group(1))
    polarity = 1 if match.group(2).upper() == "ON" else 0
    base_time = frame_number * time_delta
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise Exception()
                
    # Find white pixels (events)
    # Assuming white pixels (255) represent events, black pixels (0) represent no events
    event_pixels = np.where(image > 128)  # Threshold for white pixels
    
    y_coords = event_pixels[0].astype(np.uint64)
    x_coords = event_pixels[1].astype(np.uint64)
    
    timestamps = np.full(len(x_coords), base_time, dtype=np.uint32)
    polarities = np.full(len(x_coords), polarity, dtype=np.uint8)
    
    print(f"Extracted {len(x_coords)} events from frame {frame_number} (polarity: {polarity}, timestamp: {base_time})")
    
    return x_coords, y_coords, timestamps, polarities

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Frame to Event Converter')
    parser.add_argument('base-dir', help='Base directory to search for dynamic folders')
    parser.add_argument('folder-id', help='Folder ID for dynamic folder detection (e.g., "rec")')
    parser.add_argument('output', help='Output text eventlog file (.txt)')
    parser.add_argument('heartbeat-port', type=int, help='Port for heartbeat communication')
    parser.add_argument('command-port', type=int, help='Port for receiving commands from supervisor')
    parser.add_argument('--time-delta', type=int, default=300, help='Discrete time delta between frames in milliseconds')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing (default: 10)')
    parser.add_argument('--height', type=int, default=256, help='Image height')
    parser.add_argument('--width', type=int, default=256, help='Image width')

    args = parser.parse_args()

    if not os.path.exists(args.base_dir):
        print(f"Base directory does not exist: {args.base_dir}")
        sys.exit(1)

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    shutdown_requested = False
    
    def shutdown_callback():
        nonlocal shutdown_requested
        print("Shutdown requested by supervisor")
        shutdown_requested = True
    
    supervisor_client = SupervisorClient(args.heartbeat_port, args.command_port)
    supervisor_client.set_stop_callback(shutdown_callback)
    supervisor_client.start()

    file_queue = Queue()
    file_watcher = FileWatcher(file_queue, args.base_dir)
    file_watcher.start()
    print(f"Watching directory: {args.base_dir}")

    event_logger = EventLogger(args.output, args.height, args.width)
    
    try:
        while not shutdown_requested:
            try:
                file_path = file_queue.get(timeout=1.0)
                if file_path is None:
                    time.sleep(0.05)
                    continue

                x, y, t, p = extract_events_from_image(file_path, args.time_delta)
                event_logger.append_events(x, y, t, p)

            except Empty:
                print(f"Empty queue. Waiting for file...")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error processing file: {e}")
            finally:
                time.sleep(0.05)
                
    except KeyboardInterrupt:
        print("Received keyboard interrupt")
    finally:
        print("Shutting down gracefully...")
        supervisor_client.stop()
        file_watcher.stop()
        event_logger.close()
        print("Shutdown complete")

if __name__ == "__main__":
    main()
