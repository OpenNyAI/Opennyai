# """
# Utilities for working with the local dataset cache.
# """
# import os
# import logging
# from os import PathLike
# from pathlib import Path
# from typing import (
#     Optional,
#     Tuple,
#     Union,
#     Callable,
#     Set,
#     List,
#     Iterator,
#     Iterable,
#     Dict,
#     NamedTuple,
#     MutableMapping,
# )
#
# import cached_path as _cached_path
# from cached_path import (  # noqa: F401
#     resource_to_filename as _resource_to_filename,
#     check_tarfile,
#     is_url_or_existing_file,
#     find_latest_cached as _find_latest_cached,
# )
#
#
#
# logger = logging.getLogger(__name__)
#
# CACHE_ROOT = Path(os.getenv("ALLENNLP_CACHE_ROOT", Path.home() / ".allennlp"))
# CACHE_DIRECTORY = str(CACHE_ROOT / "cache")
# DEPRECATED_CACHE_DIRECTORY = str(CACHE_ROOT / "datasets")
#
# # This variable was deprecated in 0.7.2 since we use a single folder for caching
# # all types of files (datasets, models, etc.)
# DATASET_CACHE = CACHE_DIRECTORY
#
# # Warn if the user is still using the deprecated cache directory.
# if os.path.exists(DEPRECATED_CACHE_DIRECTORY):
#     logger.warning(
#         f"Deprecated cache directory found ({DEPRECATED_CACHE_DIRECTORY}).  "
#         f"Please remove this directory from your system to free up space."
#     )
#
#
#
#
# def cached_path(
#     url_or_filename: Union[str, PathLike],
#     cache_dir: Union[str, Path] = None,
#     extract_archive: bool = False,
#     force_extract: bool = False,
# ) -> str:
#     """
#     Given something that might be a URL or local path, determine which.
#     If it's a remote resource, download the file and cache it, and
#     then return the path to the cached file. If it's already a local path,
#     make sure the file exists and return the path.
#
#     For URLs, "http://", "https://", "s3://", "gs://", and "hf://" are all supported.
#     The latter corresponds to the HuggingFace Hub.
#
#     For example, to download the PyTorch weights for the model `epwalsh/bert-xsmall-dummy`
#     on HuggingFace, you could do:
#
#     ```python
#     cached_path("hf://epwalsh/bert-xsmall-dummy/pytorch_model.bin")
#     ```
#
#     For paths or URLs that point to a tarfile or zipfile, you can also add a path
#     to a specific file to the `url_or_filename` preceeded by a "!", and the archive will
#     be automatically extracted (provided you set `extract_archive` to `True`),
#     returning the local path to the specific file. For example:
#
#     ```python
#     cached_path("model.tar.gz!weights.th", extract_archive=True)
#     ```
#
#     # Parameters
#
#     url_or_filename : `Union[str, Path]`
#         A URL or path to parse and possibly download.
#
#     cache_dir : `Union[str, Path]`, optional (default = `None`)
#         The directory to cache downloads.
#
#     extract_archive : `bool`, optional (default = `False`)
#         If `True`, then zip or tar.gz archives will be automatically extracted.
#         In which case the directory is returned.
#
#     force_extract : `bool`, optional (default = `False`)
#         If `True` and the file is an archive file, it will be extracted regardless
#         of whether or not the extracted directory already exists.
#
#         !!! Warning
#             Use this flag with caution! This can lead to race conditions if used
#             from multiple processes on the same file.
#     """
#     import pdb;
#     return str(
#         _cached_path.cached_path(
#             url_or_filename,
#             cache_dir=cache_dir or CACHE_DIRECTORY,
#             extract_archive=extract_archive,
#             force_extract=force_extract,
#         )
#     )
#
