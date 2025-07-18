import argparse
import os
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from event_loader import EventDataLoader, EventLog, VideoMetadata
from typing import List

@jit(nopython=True)
def deduplicate_events(log: EventLog):    
    x, y, t, p = log

    # Even though events are kinda sorted by time in the event log,
    # I do sort once again just be 100% sure.
    # Either way, if the log is indeed sorted, it would take only O(n) time to check.
    sort_idx = np.argsort(t)
    sorted_t = t[sort_idx]
    sorted_x = x[sort_idx]
    sorted_y = y[sort_idx]
    sorted_p = p[sort_idx]

    unique_mask = np.ones(len(sorted_x), dtype=np.bool_)
    for i in range(1, len(sorted_x)):
        # Ignoring polality for now
        if sorted_x[i] == sorted_x[i - 1] and sorted_y[i] == sorted_y[i - 1] and sorted_t[i] == sorted_t[i - 1]:
            unique_mask[i] = False

    print("Number of events before deduplication: ", len(x))
    print("Number of events after deduplication: ", len(sorted_x[unique_mask]))
    print("Percentage of duplicates: ", (len(x) - len(sorted_x[unique_mask])) / len(x) * 100)

    return sorted_x[unique_mask], sorted_y[unique_mask], sorted_t[unique_mask], sorted_p[unique_mask]


@jit(nopython=True)
def calculate_collision_chance_for_delta_t_frames(log: EventLog, video_metadata: VideoMetadata, delta_t: int):
    height, width, start_time, end_time = video_metadata
    total_frames = ((end_time - start_time) // delta_t) + 1

    x, y, t, _ = log

    # Aggregate per-frame pixel events
    # Within the same timeframe [T1;T2] and within the same pixel (X, Y), all events must get the same index
    pixel_idx = y * width + x
    frame_idx = (t - start_time) // delta_t
    pixel_inside_frame_idx = frame_idx * width * height + pixel_idx

    # Create an index mapping to avoid excessive memory usage
    sort_idx = np.argsort(pixel_inside_frame_idx)
    sorted_pixel_inside_frame_idx = pixel_inside_frame_idx[sort_idx]

    # Example
    # Input:      [0, 0, 4, 4, 4, 5, 6, 6]
    # Input[1:]:  [0, 4, 4, 4, 5, 6, 6]     Shifted to the left by 1
    # Input[:-1]: [0, 0, 4, 4, 4, 5, 6]     Shifted to the right by 1
    # Boundary:   [F, T, F, F, T, T, F]
    # Idx:        [1, 4, 5]
    boundary_idx = np.where(sorted_pixel_inside_frame_idx[1:] != sorted_pixel_inside_frame_idx[:-1])[0]

    # Make nice to use indices
    # Example
    # Input:      [0, 0, 4, 4, 4, 5, 6, 6]
    # Boundary:   [F, T, F, F, T, T, F]
    # Idx:        [1, 4, 5]
    # Desired:    [0;1], [2;4], [5], [6;7]
    # Start:      [0, 2, 5, 6]
    # End:        [1, 4, 5, 7]
    start_indices = np.concatenate((np.array([0]), boundary_idx + 1))
    end_indices = np.concatenate((boundary_idx + 1, np.array([len(sorted_pixel_inside_frame_idx)])))

    total_active_pixels = np.zeros(total_frames, dtype=np.int32)
    total_collision_pixels = np.zeros(total_frames, dtype=np.int32)

    for i in range(len(start_indices)):
        start_group = start_indices[i]
        end_group = end_indices[i]
        
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


# np.mean is not jit-able? So we do it in raw python function
def calculate_mean_collision_chance_for_delta_t(log: EventLog, video_metadata: VideoMetadata, delta_t: int):
    mean_collision_rate_per_frames = np.array(calculate_collision_chance_for_delta_t_frames(log, video_metadata, delta_t))
    return np.mean(mean_collision_rate_per_frames) * 100

def plot_analysis_results(collision_chances: List[float], deduplicated_collision_chances: List[float], delta_t_values: List[int]):
    """
    Create visualization plots for all analysis results.
    """    

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    ax[0].plot(delta_t_values, collision_chances, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Δt (μs)')
    ax[0].set_ylabel('Collision Chance')
    ax[0].set_title('Collision Chance vs Δt')
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(delta_t_values, deduplicated_collision_chances, 'o-', linewidth=2, markersize=8)
    ax[1].set_xlabel('Δt (μs)')
    ax[1].set_ylabel('Collision Chance')
    ax[1].set_title('Deduplicated Collision Chance vs Δt')
    ax[1].grid(True, alpha=0.3)

    return plt

def main():
    parser = argparse.ArgumentParser(description='DVS Simulation Analysis')
    parser.add_argument('h5_file', help='Path to HDF5 event log file')
    parser.add_argument('--delta-t', nargs='+', type=int, default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000], 
                       help='Delta-t values to test (default: 10 to 100 μs)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.h5_file):
        print(f"❌ Error: File '{args.h5_file}' not found")
        return
    
    with h5py.File(args.h5_file, 'r') as h5_file:
        event_loader = EventDataLoader(h5_file)
        event_loader.print_metadata()
        
        event_log = event_loader.query_timerange(event_loader.get_start_time(), event_loader.get_end_time())
        deduplicated_event_log = deduplicate_events(event_log)

        video_metadata = event_loader.get_video_metadata()

        pbar = tqdm(args.delta_t, desc="Analyzing collision chance", unit="Δt", 
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        collision_chances = []
        deduplicated_collision_chances = []

        for delta_t in pbar:
            pbar.set_postfix_str(f"Δt={delta_t}μs")

            mean_collision_chance = calculate_mean_collision_chance_for_delta_t(event_log, video_metadata, delta_t)
            deduplicated_collision_chance = calculate_mean_collision_chance_for_delta_t(deduplicated_event_log, video_metadata, delta_t)
            collision_chances.append(mean_collision_chance)
            deduplicated_collision_chances.append(deduplicated_collision_chance)

        plt = plot_analysis_results(collision_chances, deduplicated_collision_chances, args.delta_t)

        filename = os.path.basename(args.h5_file)
        output_filename = filename.replace('.h5', '_eventlog_analysis.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    main()
