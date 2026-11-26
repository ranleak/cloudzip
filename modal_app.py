"""Root Modal.com deployment entrypoint for CloudZIP (`modal serve modal_app.py` / `modal deploy modal_app.py`).

Copyright (c) 2026 Alexander Scott. All rights reserved.
"""

from cloudzip.modal_app import CloudZipGpuWorker, app, image, main, web_api

__all__ = ["CloudZipGpuWorker", "app", "image", "main", "web_api"]
