#!/usr/bin/env python3
"""
Super S3 Uploader
-----------------

Features:
- Upload a single file OR a whole directory (recursive) to S3.
- Preserves relative paths inside the directory.
- Optional timestamping:
    - Add timestamp to filenames (output_20251211_235959.mp4), and/or
    - Add timestamp to the prefix (backups/20251211_235959/…)
- Robustness:
    - Retries with backoff on transient errors.
    - Per-file error handling with final summary + proper exit codes.
    - Pre-flight bucket + credentials check.
- Performance:
    - Multipart uploads via TransferConfig.
    - Parallel uploads for directories.
- Safety:
    - Overwrite policy: overwrite / skip_if_exists / fail_if_exists.
    - Include / exclude patterns.
    - Optional guards for max total bytes / file count.
- Extras:
    - Server-side encryption, storage class, optional ACL & metadata.
    - Log level flags: --verbose / --quiet.
    - Dry-run + list-only mode with size summary.
    - Optional JSON summary output for automation.

Usage examples:
    python s3_uploader.py
    python s3_uploader.py --path ./logs
    python s3_uploader.py --path outputs/video/output.mp4 --timestamp-filename
    python s3_uploader.py --path ./backup --timestamp-prefix --overwrite-policy skip_if_exists

Precedence for config:
    CLI > Environment variables > Config dataclass defaults
"""
from __future__ import annotations

# =====================================================================
# IMPORT-SAFE MODE FOR PIP
# =====================================================================
# pip import runs "egg_info" with no arguments → ANY output breaks install.
# We suppress all top-level initialization during module import.
_S3_IMPORT_SAFE = False

import argparse
import fnmatch
import json
import logging
import mimetypes
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, List, Dict, Tuple, Optional

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError, BotoCoreError


# =====================================================================
# CONFIG "KNOBS" – EDIT THESE FOR YOUR SETUP
# =====================================================================


@dataclass
class Config:
    # --- S3 target ---
    BUCKET_NAME: str = "ubada-backup-bucket"
    REGION: str = "eu-north-1"  # Your bucket region
    DEFAULT_S3_PREFIX: str = "backups"  # "folder" inside bucket (no leading slash)

    # --- AWS auth ---
    AWS_PROFILE: Optional[str] = "ubada-s3"

    # --- Local path to upload (file or directory) ---
    LOCAL_PATH: str = "./outputs/video/output.mp4"

    # --- Behavior toggles ---
    DRY_RUN: bool = False  # Log what would be uploaded, but don't actually upload
    LIST_ONLY: bool = False  # Just list what would be uploaded (no S3 needed)

    # --- Timestamps / naming ---
    ADD_TIMESTAMP_TO_FILENAME: bool = False
    TIMESTAMP_IN_PREFIX: bool = False
    TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M%S"
    USE_UTC_TIMESTAMP: bool = True

    # --- Overwrite behaviour ---
    #   "overwrite" | "skip_if_exists" | "fail_if_exists"
    OVERWRITE_POLICY: str = "overwrite"

    # --- Include / exclude patterns (fnmatch, applied to relative paths) ---
    INCLUDE_PATTERNS: List[str] = field(default_factory=lambda: ["**"])
    EXCLUDE_PATTERNS: List[str] = field(
        default_factory=lambda: ["**/__pycache__/**", "**/*.tmp", ".DS_Store"]
    )

    # --- Upload performance options (for big files) ---
    MULTIPART_THRESHOLD_MB: int = 64  # multipart starts above this size
    MAX_CONCURRENCY: int = 8  # threads for multipart upload
    DIR_UPLOAD_CONCURRENCY: int = 4  # parallel uploads when walking directory

    # --- Reliability / retry ---
    MAX_RETRIES: int = 3
    BACKOFF_SECONDS: float = 2.0  # base backoff (exponential: base * 2^attempt)

    # --- Safety limits (0 or None = disabled) ---
    MAX_TOTAL_BYTES: int = 0
    MAX_FILE_COUNT: int = 0

    # --- S3 options ---
    USE_SSE: bool = False
    SSE_TYPE: str = "AES256"  # "AES256" or "aws:kms"
    SSE_KMS_KEY_ID: Optional[str] = None
    STORAGE_CLASS: str = "STANDARD"
    ACL: Optional[str] = None  # e.g. "private", "public-read"

    # Custom metadata (key -> value) to attach
    METADATA: Dict[str, str] = field(default_factory=dict)

    # --- Progress / UX ---
    ENABLE_PROGRESS: bool = True
    JSON_SUMMARY_PATH: Optional[str] = None  # e.g. "./upload_summary.json"


# =====================================================================
# LOGGING SETUP
# =====================================================================


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("s3_uploader")
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


# =====================================================================
# Initialize logger and config
# =====================================================================
def _lazy_init():
    """Initialize CONFIG and logger ONLY when running as CLI, not on import."""
    global CONFIG, logger
    CONFIG = Config()
    logger = setup_logger()


# Only initialize when not imported by pip build system
if not _S3_IMPORT_SAFE:
    _lazy_init()


# =====================================================================
# HELPER CLASSES
# =====================================================================


class ProgressTracker:
    def __init__(self, filename: str, filesize: int) -> None:
        self.filename = filename
        self.filesize = filesize
        self._seen = 0
        self._last_logged_ratio = -1

    def __call__(self, bytes_amount: int) -> None:
        if self.filesize <= 0:
            return
        self._seen += bytes_amount
        ratio = int((self._seen / self.filesize) * 10)  # 0–10 steps (10% increments)
        if ratio != self._last_logged_ratio:
            self._last_logged_ratio = ratio
            logger.debug(
                "Progress %s: %d%% (%d/%d bytes)",
                self.filename,
                ratio * 10,
                self._seen,
                self.filesize,
            )


# =====================================================================
# CORE LOGIC
# =====================================================================


def get_effective_timestamp(cfg: Config) -> str:
    now = datetime.now(timezone.utc if cfg.USE_UTC_TIMESTAMP else None)
    return now.strftime(cfg.TIMESTAMP_FORMAT)


def create_boto3_session(profile: Optional[str], region: str) -> boto3.Session:
    """
    Create a boto3 Session using optional profile and explicit region.
    """
    session_kwargs = {"region_name": region}
    if profile:
        session_kwargs["profile_name"] = profile
        logger.info("Using AWS profile: %s", profile)
    else:
        logger.info("Using default AWS credentials (env vars / default profile / role)")

    try:
        session = boto3.Session(**session_kwargs)
    except Exception as exc:  # very early error
        logger.error("Failed to create boto3 Session: %s", exc)
        raise
    return session


def preflight_check(s3_client, bucket: str) -> None:
    """
    Ensure bucket exists and credentials are valid.
    """
    try:
        s3_client.head_bucket(Bucket=bucket)
        logger.info("Pre-flight check OK: bucket %s is accessible.", bucket)
    except (ClientError, BotoCoreError) as exc:
        logger.error("Pre-flight check failed for bucket %s: %s", bucket, exc)
        raise


def guess_content_type(path: str) -> Dict[str, str]:
    """
    Guess ContentType for nicer behavior in browsers.
    """
    content_type, _ = mimetypes.guess_type(path)
    return {"ContentType": content_type} if content_type else {}


def build_extra_args(cfg: Config, local_path: str) -> Dict[str, object]:
    extra: Dict[str, object] = {}
    extra.update(guess_content_type(local_path))

    # Storage class
    if cfg.STORAGE_CLASS:
        extra["StorageClass"] = cfg.STORAGE_CLASS

    # ACL (opt-in)
    if cfg.ACL:
        extra["ACL"] = cfg.ACL

    # Encryption
    if cfg.USE_SSE:
        if cfg.SSE_TYPE == "aws:kms":
            extra["ServerSideEncryption"] = "aws:kms"
            if cfg.SSE_KMS_KEY_ID:
                extra["SSEKMSKeyId"] = cfg.SSE_KMS_KEY_ID
        else:
            extra["ServerSideEncryption"] = "AES256"

    # Metadata
    if cfg.METADATA:
        extra["Metadata"] = cfg.METADATA

    return extra


def key_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in (
            "404",
            "NoSuchKey",
            "NotFound",
        ):
            return False
        raise


def apply_overwrite_policy(
    cfg: Config,
    s3_client,
    bucket: str,
    key: str,
) -> bool:
    """
    Decide whether to proceed with upload based on overwrite policy.
    Returns True if we should upload, False if we should skip.
    Raises RuntimeError if policy is fail_if_exists and key exists.
    """
    if cfg.OVERWRITE_POLICY == "overwrite":
        return True

    if not key_exists(s3_client, bucket, key):
        return True

    if cfg.OVERWRITE_POLICY == "skip_if_exists":
        logger.info(
            "Skipping existing object (overwrite policy): s3://%s/%s", bucket, key
        )
        return False

    if cfg.OVERWRITE_POLICY == "fail_if_exists":
        msg = f"Object already exists and overwrite policy=fail_if_exists: s3://{bucket}/{key}"
        logger.error(msg)
        raise RuntimeError(msg)

    # Unknown policy – be safe and refuse
    msg = f"Unknown OVERWRITE_POLICY={cfg.OVERWRITE_POLICY!r}"
    logger.error(msg)
    raise RuntimeError(msg)


def upload_file_once(
    cfg: Config,
    s3_client,
    bucket: str,
    local_path: str,
    s3_key: str,
    transfer_config: TransferConfig,
    run_timestamp: str,
) -> None:
    """
    Upload a single file to S3 (single attempt, no retries).
    """

    logger.info("Uploading file: %s -> s3://%s/%s", local_path, bucket, s3_key)

    extra_args = build_extra_args(cfg, local_path)

    filesize = os.path.getsize(local_path)
    callback = None
    if cfg.ENABLE_PROGRESS and logger.isEnabledFor(logging.DEBUG):
        callback = ProgressTracker(os.path.basename(local_path), filesize)

    s3_client.upload_file(
        Filename=local_path,
        Bucket=bucket,
        Key=s3_key,
        ExtraArgs=extra_args or None,
        Config=transfer_config,
        Callback=callback,
    )

    logger.info("✅ Upload success: %s -> s3://%s/%s", local_path, bucket, s3_key)


def upload_file_with_retries(
    cfg: Config,
    s3_client,
    bucket: str,
    local_path: str,
    s3_key: str,
    transfer_config: TransferConfig,
    run_timestamp: str,
) -> None:
    """
    Upload a single file with retry + backoff.
    """
    attempt = 0
    while True:
        try:
            upload_file_once(
                cfg=cfg,
                s3_client=s3_client,
                bucket=bucket,
                local_path=local_path,
                s3_key=s3_key,
                transfer_config=transfer_config,
                run_timestamp=run_timestamp,
            )
            return
        except (ClientError, BotoCoreError, OSError) as exc:
            attempt += 1
            if attempt > cfg.MAX_RETRIES:
                logger.error(
                    "❌ Upload failed permanently after %d attempts for %s: %s",
                    attempt,
                    local_path,
                    exc,
                )
                raise
            backoff = cfg.BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Upload failed for %s (attempt %d/%d). Retrying in %.1f seconds. Error: %s",
                local_path,
                attempt,
                cfg.MAX_RETRIES,
                backoff,
                exc,
            )
            time.sleep(backoff)


def walk_directory(root_path: str) -> Iterable[str]:
    """
    Yield all files under root_path (recursive).
    """
    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            yield os.path.join(dirpath, fname)


def matches_patterns(
    path: str, include_patterns: List[str], exclude_patterns: List[str]
) -> bool:
    """
    Decide if a relative path should be included based on include/exclude patterns.
    """
    # Normalize path to forward slashes
    norm = path.replace("\\", "/")

    if include_patterns:
        included = any(fnmatch.fnmatch(norm, pat) for pat in include_patterns)
        if not included:
            return False

    if exclude_patterns:
        if any(fnmatch.fnmatch(norm, pat) for pat in exclude_patterns):
            return False

    return True


def apply_timestamp_to_filename(filename: str, run_timestamp: str) -> str:
    """
    Insert timestamp before the extension: output.mp4 -> output_YYYYMMDD_HHMMSS.mp4
    """
    stem, ext = os.path.splitext(filename)
    return f"{stem}_{run_timestamp}{ext}"


def build_s3_key_for_file(
    cfg: Config,
    rel_path: str,
    base_prefix: str,
    is_single_file: bool,
    run_timestamp: str,
) -> str:
    """
    Build final S3 key for a file, considering prefix + timestamp config.
    """
    rel_path = rel_path.replace("\\", "/")

    # Filename-level timestamp
    if cfg.ADD_TIMESTAMP_TO_FILENAME:
        if is_single_file:
            rel_path = apply_timestamp_to_filename(rel_path, run_timestamp)
        else:
            # For directories, timestamp only the leaf name
            dir_part, leaf = os.path.split(rel_path)
            leaf_ts = apply_timestamp_to_filename(leaf, run_timestamp)
            rel_path = f"{dir_part}/{leaf_ts}" if dir_part else leaf_ts

    prefix = base_prefix.strip("/")
    if cfg.TIMESTAMP_IN_PREFIX:
        # Put everything under prefix/timestamp/
        parts = [p for p in [prefix, run_timestamp] if p]
        prefix_str = "/".join(parts)
    else:
        prefix_str = prefix

    prefix_str = f"{prefix_str}/" if prefix_str else ""
    return f"{prefix_str}{rel_path}"


def collect_files(
    cfg: Config,
    local_path: str,
) -> Tuple[List[Tuple[str, str]], int]:
    """
    Collect all files and their relative paths.
    Returns ([(absolute_path, relative_path), ...], total_bytes)
    """
    local_path = os.path.abspath(local_path)

    if os.path.isfile(local_path):
        root_dir = os.path.dirname(local_path)
        rel = os.path.basename(local_path)
        files = [(local_path, rel)]
    elif os.path.isdir(local_path):
        root_dir = local_path
        files: List[Tuple[str, str]] = []
        for file_path in walk_directory(root_dir):
            rel_path = os.path.relpath(file_path, root_dir)
            rel_path = rel_path.replace("\\", "/")
            if not matches_patterns(
                rel_path, cfg.INCLUDE_PATTERNS, cfg.EXCLUDE_PATTERNS
            ):
                logger.debug("Skipping by pattern: %s", rel_path)
                continue
            files.append((os.path.abspath(file_path), rel_path))
    else:
        raise RuntimeError(f"Path is neither file nor directory: {local_path}")

    total_bytes = sum(os.path.getsize(p) for p, _ in files)

    return files, total_bytes


def enforce_safety_limits(cfg: Config, total_bytes: int, file_count: int) -> None:
    if cfg.MAX_TOTAL_BYTES and total_bytes > cfg.MAX_TOTAL_BYTES:
        raise RuntimeError(
            f"Safety limit exceeded: total size {total_bytes} bytes "
            f"> MAX_TOTAL_BYTES={cfg.MAX_TOTAL_BYTES}"
        )

    if cfg.MAX_FILE_COUNT and file_count > cfg.MAX_FILE_COUNT:
        raise RuntimeError(
            f"Safety limit exceeded: file count {file_count} "
            f"> MAX_FILE_COUNT={cfg.MAX_FILE_COUNT}"
        )


def upload_path(
    cfg: Config,
    session: Optional[boto3.Session],
    bucket: str,
    local_path: str,
    base_prefix: str,
    run_timestamp: str,
) -> Tuple[int, int, List[Dict[str, str]], int]:
    """
    Upload a file or directory to S3.
    Returns (success_count, fail_count, failures_list, total_bytes).
    """
    files, total_bytes = collect_files(cfg, local_path)
    file_count = len(files)

    logger.info("Discovered %d files, total size %d bytes", file_count, total_bytes)
    enforce_safety_limits(cfg, total_bytes, file_count)

    if cfg.LIST_ONLY or cfg.DRY_RUN:
        logger.info(
            "[DRY/LIST] Would upload %d files (%d bytes) to s3://%s/%s",
            file_count,
            total_bytes,
            bucket,
            base_prefix,
        )
        for abs_path, rel_path in files:
            is_single = os.path.isfile(os.path.abspath(local_path))
            s3_key = build_s3_key_for_file(
                cfg=cfg,
                rel_path=rel_path,
                base_prefix=base_prefix,
                is_single_file=is_single,
                run_timestamp=run_timestamp,
            )
            logger.info("[DRY/LIST] %s -> s3://%s/%s", abs_path, bucket, s3_key)
        return 0, 0, [], total_bytes

    if session is None:
        raise RuntimeError("Session must not be None when not in dry/list mode.")

    s3_client = session.client("s3")
    preflight_check(s3_client, bucket)

    transfer_config = TransferConfig(
        multipart_threshold=cfg.MULTIPART_THRESHOLD_MB * 1024 * 1024,
        max_concurrency=cfg.MAX_CONCURRENCY,
        multipart_chunksize=8 * 1024 * 1024,
        use_threads=True,
    )

    success_count = 0
    fail_count = 0
    failures: List[Dict[str, str]] = []

    is_single = os.path.isfile(os.path.abspath(local_path))

    # Parallel uploads for directories
    if file_count > 1 and cfg.DIR_UPLOAD_CONCURRENCY > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def worker(abs_path: str, rel_path: str) -> Tuple[str, Optional[str]]:
            try:
                s3_key = build_s3_key_for_file(
                    cfg=cfg,
                    rel_path=rel_path,
                    base_prefix=base_prefix,
                    is_single_file=is_single,
                    run_timestamp=run_timestamp,
                )

                if not apply_overwrite_policy(cfg, s3_client, bucket, s3_key):
                    return abs_path, None  # skipped

                upload_file_with_retries(
                    cfg=cfg,
                    s3_client=s3_client,
                    bucket=bucket,
                    local_path=abs_path,
                    s3_key=s3_key,
                    transfer_config=transfer_config,
                    run_timestamp=run_timestamp,
                )
                return abs_path, None
            except Exception as exc:
                return abs_path, str(exc)

        with ThreadPoolExecutor(max_workers=cfg.DIR_UPLOAD_CONCURRENCY) as executor:
            future_map = {
                executor.submit(worker, abs_path, rel_path): (abs_path, rel_path)
                for abs_path, rel_path in files
            }

            for future in as_completed(future_map):
                abs_path, rel_path = future_map[future]
                _, err = future.result()
                if err is None:
                    success_count += 1
                else:
                    fail_count += 1
                    failures.append(
                        {
                            "local_path": abs_path,
                            "relative_path": rel_path,
                            "error": err,
                        }
                    )
    else:
        # Single-threaded (single file or concurrency disabled)
        for abs_path, rel_path in files:
            try:
                s3_key = build_s3_key_for_file(
                    cfg=cfg,
                    rel_path=rel_path,
                    base_prefix=base_prefix,
                    is_single_file=is_single,
                    run_timestamp=run_timestamp,
                )

                if not apply_overwrite_policy(cfg, s3_client, bucket, s3_key):
                    continue

                upload_file_with_retries(
                    cfg=cfg,
                    s3_client=s3_client,
                    bucket=bucket,
                    local_path=abs_path,
                    s3_key=s3_key,
                    transfer_config=transfer_config,
                    run_timestamp=run_timestamp,
                )
                success_count += 1
            except Exception as exc:
                fail_count += 1
                failures.append(
                    {
                        "local_path": abs_path,
                        "relative_path": rel_path,
                        "error": str(exc),
                    }
                )

    return success_count, fail_count, failures, total_bytes


# =====================================================================
# CLI / ENTRYPOINT
# =====================================================================


def parse_args() -> argparse.Namespace:
    # Env defaults
    env_bucket = os.getenv("S3U_BUCKET_NAME", CONFIG.BUCKET_NAME)
    env_prefix = os.getenv("S3U_PREFIX", CONFIG.DEFAULT_S3_PREFIX)
    env_region = os.getenv("S3U_REGION", CONFIG.REGION)
    env_profile = os.getenv("S3U_PROFILE", CONFIG.AWS_PROFILE or "") or None
    env_local_path = os.getenv("S3U_LOCAL_PATH", CONFIG.LOCAL_PATH)

    parser = argparse.ArgumentParser(
        description="Upload a file or directory to an S3 bucket using boto3.",
        epilog="""
Examples:
  s3_uploader.py
  s3_uploader.py --path ./logs --timestamp-prefix
  s3_uploader.py --path output.mp4 --timestamp-filename --overwrite-policy skip_if_exists
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--path",
        "-p",
        dest="local_path",
        default=env_local_path,
        help=f"Local file or directory to upload (default: {env_local_path})",
    )
    parser.add_argument(
        "--bucket",
        "-b",
        dest="bucket",
        default=env_bucket,
        help=f"S3 bucket name (default: {env_bucket})",
    )
    parser.add_argument(
        "--prefix",
        "-x",
        dest="prefix",
        default=env_prefix,
        help=f"S3 prefix/folder inside the bucket (default: {env_prefix})",
    )
    parser.add_argument(
        "--region",
        "-r",
        dest="region",
        default=env_region,
        help=f"AWS region (default: {env_region})",
    )
    parser.add_argument(
        "--profile",
        dest="profile",
        default=env_profile,
        help=f"AWS profile name (default: {env_profile})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be uploaded without actually uploading.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List files that would be uploaded (no AWS calls). Implies --dry-run.",
    )
    parser.add_argument(
        "--timestamp-filename",
        action="store_true",
        help="Append timestamp to filenames (before extension).",
    )
    parser.add_argument(
        "--timestamp-prefix",
        action="store_true",
        help="Append timestamp as a folder in the prefix.",
    )
    parser.add_argument(
        "--overwrite-policy",
        choices=["overwrite", "skip_if_exists", "fail_if_exists"],
        default=CONFIG.OVERWRITE_POLICY,
        help=f"How to handle existing keys (default: {CONFIG.OVERWRITE_POLICY})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Increase logging verbosity to DEBUG.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Reduce logging verbosity to WARNING.",
    )
    parser.add_argument(
        "--json-summary",
        dest="json_summary",
        default=CONFIG.JSON_SUMMARY_PATH,
        help="Optional path to write JSON summary of the upload.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Super S3 Uploader 1.0.0",
    )

    return parser.parse_args()


def write_json_summary(
    path: Optional[str],
    bucket: str,
    prefix: str,
    success_count: int,
    fail_count: int,
    failures: List[Dict[str, str]],
    total_bytes: int,
    started_at: float,
    finished_at: float,
) -> None:
    if not path:
        return

    summary = {
        "bucket": bucket,
        "prefix": prefix,
        "success_count": success_count,
        "fail_count": fail_count,
        "failures": failures,
        "total_bytes": total_bytes,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": finished_at - started_at,
        "host": platform.node(),
        "os": platform.platform(),
        "python": sys.version,
    }

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Wrote JSON summary to %s", path)
    except OSError as exc:
        logger.error("Failed to write JSON summary to %s: %s", path, exc)


def main() -> int:
    args = parse_args()

    # Adjust logger level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)

    # Build runtime config from defaults + CLI
    cfg = CONFIG
    cfg.DRY_RUN = bool(args.dry_run or args.list_only)
    cfg.LIST_ONLY = bool(args.list_only)
    cfg.ADD_TIMESTAMP_TO_FILENAME = bool(args.timestamp_filename)
    cfg.TIMESTAMP_IN_PREFIX = bool(args.timestamp_prefix)
    cfg.OVERWRITE_POLICY = args.overwrite_policy
    cfg.JSON_SUMMARY_PATH = args.json_summary or None

    local_path = args.local_path
    bucket = args.bucket
    prefix = args.prefix
    region = args.region
    profile = args.profile

    logger.info("=== S3 Uploader starting ===")
    logger.info("Local path         : %s", local_path)
    logger.info("Bucket             : %s", bucket)
    logger.info("Prefix             : %s", prefix)
    logger.info("Region             : %s", region)
    logger.info("Profile            : %s", profile or "<default>")
    logger.info("Dry run            : %s", cfg.DRY_RUN)
    logger.info("List only          : %s", cfg.LIST_ONLY)
    logger.info("Timestamp filename : %s", cfg.ADD_TIMESTAMP_TO_FILENAME)
    logger.info("Timestamp prefix   : %s", cfg.TIMESTAMP_IN_PREFIX)
    logger.info("Overwrite policy   : %s", cfg.OVERWRITE_POLICY)

    run_timestamp = get_effective_timestamp(cfg)

    session: Optional[boto3.Session] = None
    if not cfg.LIST_ONLY and not cfg.DRY_RUN:
        try:
            session = create_boto3_session(profile, region)
        except Exception as exc:
            logger.error("Failed to create AWS session: %s", exc)
            return 1

    start_time = time.time()

    try:
        success_count, fail_count, failures, total_bytes = upload_path(
            cfg=cfg,
            session=session,
            bucket=bucket,
            local_path=local_path,
            base_prefix=prefix,
            run_timestamp=run_timestamp,
        )
    except FileNotFoundError as exc:
        logger.error("Uploader finished with error: %s", exc)
        return 1
    except Exception as exc:
        logger.error("Uploader finished with fatal error: %s", exc)
        return 1

    end_time = time.time()

    # Summary
    duration = end_time - start_time
    logger.info(
        "Summary: success=%d, failed=%d, total_bytes=%d, duration=%.2fs",
        success_count,
        fail_count,
        total_bytes,
        duration,
    )

    write_json_summary(
        path=cfg.JSON_SUMMARY_PATH,
        bucket=bucket,
        prefix=prefix,
        success_count=success_count,
        fail_count=fail_count,
        failures=failures,
        total_bytes=total_bytes,
        started_at=start_time,
        finished_at=end_time,
    )

    if cfg.DRY_RUN or cfg.LIST_ONLY:
        logger.info("=== S3 Uploader DRY/LIST run finished successfully ===")
        return 0

    if fail_count == 0:
        logger.info(
            "=== S3 Uploader finished successfully (%d files, %.2f MB) ===",
            success_count,
            total_bytes / (1024 * 1024) if total_bytes else 0,
        )
        return 0
    elif success_count > 0:
        logger.warning(
            "=== S3 Uploader finished with partial success: %d succeeded, %d failed ===",
            success_count,
            fail_count,
        )
        return 2
    else:
        logger.error("=== S3 Uploader failed: no files uploaded successfully ===")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
