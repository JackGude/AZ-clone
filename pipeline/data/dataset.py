"""Dataset and data loading utilities for the training pipeline."""

import os
import random
from typing import List, Iterator, Any

from config import DRAW_CACHE_DIR, WIN_CACHE_DIR
import torch
from torch.utils.data import IterableDataset, get_worker_info


class StreamingDataset(IterableDataset):
    """
    An IterableDataset that streams data from chunk files on disk.
    Handles worker splitting and shuffling of data.
    """

    def __init__(self, chunk_files: List[str], shuffle: bool = False):
        """Initialize the dataset.
        
        Args:
            chunk_files: List of paths to chunk files
            shuffle: Whether to shuffle the data
        """
        super().__init__()
        self.chunk_files = chunk_files
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[Any]:
        """Iterate through the dataset."""
        worker_info = get_worker_info()
        
        # Split files among workers
        if worker_info is None:  # Single-process data loading
            worker_files = self.chunk_files
            worker_id = 0
            num_workers = 1
        else:  # Multi-process data loading
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            worker_files = self.chunk_files[worker_id::num_workers]
        
        # Shuffle files if needed
        if self.shuffle:
            random.shuffle(worker_files)
        
        # Iterate through assigned chunks
        for chunk_path in worker_files:
            try:
                # Load chunk
                chunk = torch.load(chunk_path, weights_only=True)
                
                # Shuffle data points within chunk if needed
                if self.shuffle:
                    random.shuffle(chunk)
                
                # Yield each data point
                for data_point in chunk:
                    yield data_point
                    
            except Exception as e:
                print(f"Error loading chunk {chunk_path}: {e}", flush=True)
                continue


def load_tensor_files_for_training() -> List[str]:
    """
    Collect all available chunk file paths from the cache directories.

    Returns:
        List of file paths to training data chunks.
    """
    all_chunk_files = []

    # Load file paths from the decisive games cache
    if os.path.isdir(WIN_CACHE_DIR):
        win_files = [
            os.path.join(WIN_CACHE_DIR, f)
            for f in os.listdir(WIN_CACHE_DIR)
            if f.endswith(".pt")
        ]
        all_chunk_files.extend(win_files)
    else:
        print(
            f"Warning: Decisive games cache directory '{WIN_CACHE_DIR}' not found.",
            flush=True,
        )

    # Load file paths from the draw games cache
    if os.path.isdir(DRAW_CACHE_DIR):
        draw_files = [
            os.path.join(DRAW_CACHE_DIR, f)
            for f in os.listdir(DRAW_CACHE_DIR)
            if f.endswith(".pt")
        ]
        all_chunk_files.extend(draw_files)
    else:
        print(
            f"Warning: Draw games cache directory '{DRAW_CACHE_DIR}' not found.",
            flush=True,
        )

    if not all_chunk_files:
        print("No training data files found. Cannot train.", flush=True)
        return []

    return all_chunk_files
