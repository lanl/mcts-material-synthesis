"""
Data ingestion, retrieval, and external services.

Normalizes the public synthesis corpora into `RouteRecord`s (`datasets`), builds
the analog-retrieval index and precursor priors (`retrieval`), and wraps the
optional Materials Project API (`materials_project`). Heavy/optional dependencies
(`mp_api`) are imported lazily so importing this package stays lightweight.

© 2026. Triad National Security, LLC. All rights reserved.
"""

from .datasets import (
    download_public_datasets,
    load_processed_routes,
    prepare_processed_data,
)
from .retrieval import RetrievalIndex

__all__ = [
    "download_public_datasets",
    "load_processed_routes",
    "prepare_processed_data",
    "RetrievalIndex",
]
