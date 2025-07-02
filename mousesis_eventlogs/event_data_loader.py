#!/usr/bin/env python3
"""
Event Data Loader

This script loads event data from HDF5 files and provides an interface
to load events in a time range. It uses efficient HDF5 data access for fast loading.
"""

from typing import cast
import h5py
import numpy as np
import numpy.typing as npt

type EventData = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.uint64], npt.NDArray[np.uint8]]

class EventDataLoader:
    def __init__(self, h5_file: h5py.File):
        self.h5_file = h5_file
        self.height = self.h5_file['images'].shape[1]
        self.width = self.h5_file['images'].shape[2]

        self.x = cast(h5py.Dataset, self.h5_file['x'])
        self.y = cast(h5py.Dataset, self.h5_file['y'])
        self.t = cast(h5py.Dataset, self.h5_file['t'])
        self.p = cast(h5py.Dataset, self.h5_file['p'])

    def get_events_in_time_range(self, start: float, end: float) -> EventData:
        i0 = np.searchsorted(self.t, start, side='left')
        i1 = np.searchsorted(self.t, end, side='right')

        x = self.x[i0:i1]
        y = self.y[i0:i1]
        p = self.p[i0:i1]
        t = self.t[i0:i1]

        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64) 
        t = np.array(t, dtype=np.uint64)
        p = np.array(p, dtype=np.uint8)

        print(f"Number of events: {len(x)}")
        print(f"Time range: {t[0]} - {t[-1]}")

        return x, y, t, p

    def aggregate_time_range(self, start: float, end: float):
        x, y, _, p = self.get_events_in_time_range(start, end)
    
        # Create empty frame for accumulation
        frame = np.zeros((self.height, self.width), dtype=np.float32)
        
        if len(x) == 0:  # Handle empty event arrays
            return frame
        
        # Convert coordinates to integers and polarity to signed values vectorized
        x_int = x.astype(np.int32)
        y_int = y.astype(np.int32)
        p_signed = 2 * p.astype(np.int8) - 1  # Convert {0,1} to {-1,+1}
        
        # Use numpy's add.at for efficient accumulation at specified indices
        np.add.at(frame, (y_int, x_int), p_signed)
        
        # Clamp values to [-1, 1] range using vectorized numpy operation
        frame = np.clip(frame, -1, 1)
        
        return frame

    def print_metadata(self):
        print(f"File: {self.h5_file.filename}")
        print(f"Keys: {self.h5_file.keys()}")
        print(f"Number of events: {self.t.shape[0]}")
        print(f"Time range: {self.t[0]} - {self.t[-1]}")

    def get_start_time(self) -> float:
        return self.t[0]

    def get_end_time(self) -> float:
        return self.t[-1]