#!/usr/bin/env python3
"""
Duplicate Event Analysis

This script analyzes duplicate events across all sequence files (seq01.h5 to seq33.h5)
and saves the results to a text file for further analysis.
"""

import os
import h5py
import numpy as np
from tqdm import tqdm
from event_loader import EventDataLoader
from eventlog_analysis import deduplicate_events
from typing import Dict, Tuple

def analyze_duplicates_for_file(file_path: str) -> Tuple[int, int, float]:
    """
    Analyze duplicate events in a single HDF5 file.
    
    Args:
        file_path: Path to the HDF5 file
        
    Returns:
        Tuple of (total_events, unique_events, duplicate_percentage)
    """
    try:
        with h5py.File(file_path, 'r') as h5_file:
            event_loader = EventDataLoader(h5_file)
            
            event_log = event_loader.query_timerange(
                event_loader.get_start_time(), 
                event_loader.get_end_time()
            )
            
            total_events = len(event_log[0])
            
            deduplicated_event_log = deduplicate_events(event_log)            
            unique_events = len(deduplicated_event_log[0])
            
            duplicate_percentage = ((total_events - unique_events) / total_events) * 100
            
            return total_events, unique_events, duplicate_percentage
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, 0, 0.0

def main():
    dataset_dir = "dataset/mousesis/top"
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Error: Dataset directory '{dataset_dir}' not found")
        return
    
    sequence_files = []
    for i in range(1, 34):
        filename = f"seq{i:02d}.h5"
        file_path = os.path.join(dataset_dir, filename)
        if os.path.exists(file_path):
            sequence_files.append(file_path)
        else:
            print(f"⚠️  Warning: File {file_path} not found")
    
    if not sequence_files:
        print("❌ Error: No sequence files found")
        return
    
    print(f"Found {len(sequence_files)} sequence files to analyze")
    print("=" * 80)
    
    results: Dict[str, Tuple[int, int, float]] = {}
    
    pbar = tqdm(sequence_files, desc="Analyzing duplicate events", unit="file")
    
    for file_path in pbar:
        filename = os.path.basename(file_path)
        pbar.set_postfix_str(f"Processing {filename}")
        
        total_events, unique_events, duplicate_percentage = analyze_duplicates_for_file(file_path)
        results[filename] = (total_events, unique_events, duplicate_percentage)
    
    total_events_all = sum(result[0] for result in results.values())
    total_unique_events_all = sum(result[1] for result in results.values())
    overall_duplicate_percentage = ((total_events_all - total_unique_events_all) / total_events_all) * 100
    
    output_file = "duplication_analysis.txt"
    
    with open(output_file, 'w') as f:
        f.write("DUPLICATE EVENT ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Analysis Date: {os.popen('date').read().strip()}\n")
        f.write(f"Total files analyzed: {len(results)}\n")
        f.write(f"Dataset directory: {dataset_dir}\n\n")
        
        f.write("INDIVIDUAL FILE RESULTS:\n")
        f.write("-" * 30 + "\n")
        
        # Sort results by filename for consistent output
        for filename in sorted(results.keys()):
            total_events, unique_events, duplicate_percentage = results[filename]
            f.write(f"{filename}:\n")
            f.write(f"  Total events: {total_events:,}\n")
            f.write(f"  Unique events: {unique_events:,}\n")
            f.write(f"  Duplicate percentage: {duplicate_percentage:.5f}%\n")
            f.write(f"  Duplicate events: {total_events - unique_events:,}\n\n")
        
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total events across all files: {total_events_all:,}\n")
        f.write(f"Total unique events across all files: {total_unique_events_all:,}\n")
        f.write(f"Overall duplicate percentage: {overall_duplicate_percentage:.5f}%\n")
        f.write(f"Total duplicate events: {total_events_all - total_unique_events_all:,}\n\n")
        
        # Calculate per-file statistics
        duplicate_percentages = [result[2] for result in results.values()]
        f.write("PER-FILE STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Average duplicate percentage: {np.mean(duplicate_percentages):.5f}%\n")
        f.write(f"Median duplicate percentage: {np.median(duplicate_percentages):.5f}%\n")
        f.write(f"Minimum duplicate percentage: {np.min(duplicate_percentages):.5f}%\n")
        f.write(f"Maximum duplicate percentage: {np.max(duplicate_percentages):.5f}%\n")
        f.write(f"Standard deviation: {np.std(duplicate_percentages):.5f}%\n")
    
    print(f"\n✅ Analysis complete! Results saved to '{output_file}'")
    print(f"📊 Overall duplicate percentage: {overall_duplicate_percentage:.5f}%")
    print(f"📈 Average duplicate percentage per file: {np.mean(duplicate_percentages):.5f}%")

if __name__ == "__main__":
    main()
