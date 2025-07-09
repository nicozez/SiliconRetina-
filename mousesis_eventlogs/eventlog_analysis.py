import argparse
import os
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from event_data_loader import EventDataLoader, EventLog, VideoMetadata
from typing import List

@jit(nopython=True)
def calculate_mean_collision_chance_for_delta_t_frames(log: EventLog, video_metadata: VideoMetadata, delta_t: int):
    height, width, start_time, end_time = video_metadata
    total_frames = ((end_time - start_time) // delta_t) + 1

    x, y, t, _ = log

    pixel_idx = y * width + x
    frame_idx = (t - start_time) // delta_t
    pixel_inside_frame_idx = frame_idx * width * height + pixel_idx

    sort_idx = np.argsort(pixel_inside_frame_idx)
    sorted_pixel_inside_frame_idx = pixel_inside_frame_idx[sort_idx]

    boundary_idx = np.where(sorted_pixel_inside_frame_idx[1:] != sorted_pixel_inside_frame_idx[:-1])[0]
    
    total_active_pixels = np.zeros(total_frames, dtype=np.int32)
    total_collision_pixels = np.zeros(total_frames, dtype=np.int32)

    for i in range(len(boundary_idx) - 1):
        # Count events in the pixel-frame group
        start_group = boundary_idx[i] if i != 0 else -1
        end_group = boundary_idx[i + 1] if i != len(boundary_idx) - 1 else len(sorted_pixel_inside_frame_idx)

        count = end_group - start_group
        
        frame_id = frame_idx[sort_idx[start_group]]

        total_active_pixels[frame_id] += 1
        if count > 1:
            total_collision_pixels[frame_id] += 1

    mean_collision_rate_per_frame = []
    for i in range(total_frames):
        if total_active_pixels[i] > 0:
            mean_collision_rate_per_frame.append(total_collision_pixels[i] / total_active_pixels[i])
        else:
            mean_collision_rate_per_frame.append(0)

    return mean_collision_rate_per_frame

def calculate_mean_collision_chance_for_delta_t(log: EventLog, video_metadata: VideoMetadata, delta_t: int):
    mean_collision_rate_per_frames = calculate_mean_collision_chance_for_delta_t_frames(log, video_metadata, delta_t)
    return np.mean(mean_collision_rate_per_frames) * 100

def plot_analysis_results(collision_chances: List[float], delta_t_values: List[int]):
    """
    Create visualization plots for all analysis results.
    """    
    plt.plot(delta_t_values, collision_chances, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Δt (μs)')
    plt.ylabel('Collision Chance')
    plt.title('Collision Chance vs Δt')
    plt.grid(True, alpha=0.3)
    plt.show()
    

def main():
    parser = argparse.ArgumentParser(description='DVS Simulation Analysis')
    parser.add_argument('h5_file', help='Path to HDF5 event log file')
    parser.add_argument('--duration', '-d', type=float, default=500000, help='Duration in microseconds (default: 500000)')
    parser.add_argument('--delta-t', nargs='+', type=int, default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
                       help='Delta-t values to test (default: 10 to 100 μs)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.h5_file):
        print(f"❌ Error: File '{args.h5_file}' not found")
        return
    
    with h5py.File(args.h5_file, 'r') as h5_file:
        event_loader = EventDataLoader(h5_file)
        event_loader.print_metadata()
        
        event_log = event_loader.query_timerange(event_loader.get_start_time(), event_loader.get_end_time())
        video_metadata = event_loader.get_video_metadata()

        pbar = tqdm(args.delta_t, desc="Analyzing collision chance", unit="Δt", 
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        collision_chances = []
        for delta_t in pbar:
            pbar.set_postfix_str(f"Δt={delta_t}μs")

            mean_collision_chance = calculate_mean_collision_chance_for_delta_t(event_log, video_metadata, delta_t)
            collision_chances.append(mean_collision_chance)

        plot_analysis_results(collision_chances, args.delta_t)

if __name__ == "__main__":
    main()
