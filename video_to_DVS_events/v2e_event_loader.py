#!/usr/bin/env python3
"""
V2E Event Data Loader

This script loads event data from V2E HDF5 files and provides an interface
to load events in a time range. It uses efficient HDF5 data access for fast loading.
The V2E schema includes events (p, t, x, y) and frame data (images, img_ts, img2event).
"""

from typing import cast, Optional
import h5py
import numpy as np
import numpy.typing as npt

type EventData = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.uint32], npt.NDArray[np.uint8]]

class V2EEventDataLoader:
    def __init__(self, h5_file: h5py.File, height: int, width: int):
        self.h5_file = h5_file
        self.height = height
        self.width = width

        # Single events dataset: shape (N, 4) where each row is [t, x, y, p]
        self.events = cast(h5py.Dataset, self.h5_file['events'])

    def query_timerange(self, start: float, end: float) -> EventData:
        """Query events in a time range.
        
        Args:
            start: Start time in microseconds
            end: End time in microseconds
            
        Returns:
            Tuple of (x, y, t, p) arrays
        """
        # Convert microseconds to seconds for dataset query
        start_seconds = start
        end_seconds = end
        
        # Get timestamp column (first column)
        timestamps = self.events[:, 0]
        
        # Find indices for time range
        i0 = np.searchsorted(timestamps, start_seconds, side='left')
        i1 = np.searchsorted(timestamps, end_seconds, side='right')

        # Extract events in range
        events_in_range = self.events[i0:i1]
        
        if len(events_in_range) == 0:
            # Return empty arrays
            return (np.array([], dtype=np.float64),
                    np.array([], dtype=np.float64),
                    np.array([], dtype=np.uint32),
                    np.array([], dtype=np.uint8))

        # Extract columns: [t, x, y, p]
        t = events_in_range[:, 0]
        x = events_in_range[:, 1]
        y = events_in_range[:, 2]
        p = events_in_range[:, 3]

        # Convert to required types
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64) 
        t = np.array(t, dtype=np.uint32)  # Convert to microseconds
        p = np.array(p, dtype=np.uint8)

        if len(x) > 0:
            print(f"Number of events: {len(x)}")
            print(f"Time range: {t[0]} - {t[-1]} microseconds")
            print(f"Query range: {start:.0f} - {end:.0f} microseconds")
            print(f"Sample events - x: {x[:3]}, y: {y[:3]}, p: {p[:3]}")

        return x, y, t, p

    def aggregate_time_range(self, start: float, end: float) -> np.ndarray:
        """Aggregate events in a time range into a 2D frame.
        
        Args:
            start: Start time in microseconds
            end: End time in microseconds
            
        Returns:
            2D frame with aggregated events
        """
        x, y, _, p = self.query_timerange(start, end)
    
        # Create empty frame for accumulation
        frame = np.zeros((self.height, self.width), dtype=np.float32)
        
        if len(x) == 0:  # Handle empty event arrays
            return frame
        
        # Convert coordinates to integers and polarity to signed values vectorized
        x_int = x.astype(np.int32)
        y_int = y.astype(np.int32)
        
        # V2E uses 0/1 for polarity, convert to {-1,+1}
        p_signed = 2 * p.astype(np.int8) - 1  # Convert {0,1} to {-1,+1}
        
        # Clip coordinates to valid range
        x_int = np.clip(x_int, 0, self.width - 1)
        y_int = np.clip(y_int, 0, self.height - 1)
        
        # Use numpy's add.at for efficient accumulation at specified indices
        np.add.at(frame, (y_int, x_int), p_signed)
        
        # Clamp values to [-1, 1] range using vectorized numpy operation
        frame = np.clip(frame, -1, 1)
        
        print(f"Frame stats: min={frame.min():.3f}, max={frame.max():.3f}, non-zero={np.count_nonzero(frame)}")
        
        return frame


    def print_metadata(self):
        """Print dataset metadata."""
        print("=" * 80)
        print("V2E Dataset metadata:")
        print(f"File: {self.h5_file.filename}")
        print(f"Keys: {list(self.h5_file.keys())}")
        print(f"Number of events: {self.events.shape[0]}")
        print(f"Event data shape: {self.events.shape}")
        print(f"Event time range: {self.events[0, 0]:.6f} - {self.events[-1, 0]:.6f} seconds")
        print(f"Duration: {self.get_duration_seconds():.3f} seconds")
        print(f"Image dimensions: {self.height}x{self.width}")
        
        # Show sample events for debugging
        print("\nFirst 5 events [t, x, y, p]:")
        for i in range(min(5, self.events.shape[0])):
            t, x, y, p = self.events[i]
            print(f"  {i}: t={t:.6f}s, x={x:.1f}, y={y:.1f}, p={p}")
        
        print("\nEvent statistics:")
        print(f"  X range: {self.events[:, 1].min():.1f} - {self.events[:, 1].max():.1f}")
        print(f"  Y range: {self.events[:, 2].min():.1f} - {self.events[:, 2].max():.1f}")
        print(f"  P range: {self.events[:, 3].min()} - {self.events[:, 3].max()}")
        print(f"  Unique polarities: {np.unique(self.events[:, 3])}")
        print("=" * 80)

    def get_start_time(self) -> float:
        """Get the start time of events in seconds."""
        return float(self.events[0, 0])

    def get_end_time(self) -> float:
        """Get the end time of events in seconds."""
        return float(self.events[-1, 0])
    
    def get_start_time_microseconds(self) -> float:
        """Get the start time of events in microseconds."""
        return float(self.events[0, 0])

    def get_end_time_microseconds(self) -> float:
        """Get the end time of events in microseconds."""
        return float(self.events[-1, 0])

    def get_duration_microseconds(self) -> float:
        """Get the total duration in microseconds."""
        return self.get_duration_seconds()

    def get_duration_seconds(self) -> float:
        """Get the total duration in seconds."""
        return self.get_end_time() - self.get_start_time()
