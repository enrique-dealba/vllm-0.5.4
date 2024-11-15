"""Timescale Vector configuration."""

from datetime import timedelta

# Index and partitioning settings
TIME_PARTITION_INTERVAL = timedelta(hours=6)
DISKANN_INDEX_PARAMS = {
    "num_neighbors": 50,
    "search_list_size": 100,
    "max_alpha": 1.2,
    "storage_layout": "memory_optimized",
}
