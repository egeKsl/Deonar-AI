#!/usr/bin/env python3
"""
S3 Uploader
-----------

Features:
- Upload a single file OR a whole directory (recursive) to S3.
- Preserves relative paths inside the directory.
- Uses AWS credentials from:
    - AWS profile (recommended), or
    - Environment variables, or
    - Instance role (EC2/ECS/Lambda).
- Simple config "knobs" at the top + CLI overrides.

Usage examples:
    python s3_uploader.py                       # uses CONFIG.LOCAL_PATH
    python s3_uploader.py --path ./logs         # override LOCAL_PATH
    python s3_uploader.py --path backup.zip \
        --prefix backups/2025-12-08/ --profile ubada-profile
"""

from __future__ import annotations

import argparse
import logging
import mimetypes
import os
import sys
from dataclasses import dataclass

import boto3
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
    # If you use `aws configure --profile <name>`, put that name here.
    # Leave as None to use default profile / env vars / instance role.
    AWS_PROFILE: str | None = "ubada-s3"

    # --- Local path to upload (file or directory).
    # Can be overridden via CLI: --path
    LOCAL_PATH: str = "./outputs/video/output.mp4"

    # --- Behavior toggles ---
    DRY_RUN: bool = (
        False  # True = log what would be uploaded, but don't actually upload
    )

    # --- Upload performance options (for big files) ---
    MULTIPART_THRESHOLD_MB: int = 64  # multipart starts above this size
    MAX_CONCURRENCY: int = 8  # threads for upload


CONFIG = Config()


# =====================================================================
# LOGGING SETUP
# =====================================================================


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("s3_uploader")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


logger = setup_logger()


# =====================================================================
# CORE LOGIC
# =====================================================================


def create_boto3_session(profile: str | None, region: str) -> boto3.Session:
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


def guess_content_type(path: str) -> dict:
    """
    Guess ContentType for nicer behavior in browsers.
    """
    content_type, _ = mimetypes.guess_type(path)
    return {"ContentType": content_type} if content_type else {}


def upload_file(
    s3_client,
    bucket: str,
    local_path: str,
    s3_key: str,
    dry_run: bool = False,
) -> None:
    """
    Upload a single file to S3.
    """
    logger.info("Uploading file: %s -> s3://%s/%s", local_path, bucket, s3_key)

    if dry_run:
        logger.info("DRY_RUN enabled – not actually uploading.")
        return

    extra_args = guess_content_type(local_path)

    try:
        s3_client.upload_file(
            Filename=local_path,
            Bucket=bucket,
            Key=s3_key,
            ExtraArgs=extra_args or None,
        )
        logger.info("✅ Upload success: %s -> s3://%s/%s", local_path, bucket, s3_key)
    except (ClientError, BotoCoreError) as exc:
        logger.error("❌ Upload failed for %s: %s", local_path, exc)
        raise


def walk_directory(root_path: str):
    """
    Yield all files under root_path (recursive).
    """
    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            yield os.path.join(dirpath, fname)


def upload_path(
    session: boto3.Session,
    bucket: str,
    local_path: str,
    base_prefix: str = "",
    dry_run: bool = False,
) -> None:
    """
    Decide whether local_path is a file or directory and upload accordingly.
    """
    s3_client = session.client("s3")

    local_path = os.path.abspath(local_path)
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local path does not exist: {local_path}")

    base_prefix = base_prefix.strip("/")
    prefix_str = f"{base_prefix}/" if base_prefix else ""

    if os.path.isfile(local_path):
        # Single file
        file_name = os.path.basename(local_path)
        s3_key = f"{prefix_str}{file_name}"
        upload_file(s3_client, bucket, local_path, s3_key, dry_run=dry_run)

    elif os.path.isdir(local_path):
        # Directory – walk recursively
        logger.info("Detected directory. Uploading contents recursively.")
        root_dir = local_path

        for file_path in walk_directory(root_dir):
            rel_path = os.path.relpath(file_path, root_dir)
            rel_path = rel_path.replace("\\", "/")  # Windows fix
            s3_key = f"{prefix_str}{rel_path}"

            upload_file(s3_client, bucket, file_path, s3_key, dry_run=dry_run)

        logger.info("All files from directory uploaded.")
    else:
        raise RuntimeError(f"Path is neither file nor directory: {local_path}")


# =====================================================================
# CLI / ENTRYPOINT
# =====================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a file or directory to an S3 bucket using boto3."
    )

    parser.add_argument(
        "--path",
        "-p",
        dest="local_path",
        default=CONFIG.LOCAL_PATH,
        help=f"Local file or directory to upload (default: {CONFIG.LOCAL_PATH})",
    )
    parser.add_argument(
        "--bucket",
        "-b",
        dest="bucket",
        default=CONFIG.BUCKET_NAME,
        help=f"S3 bucket name (default: {CONFIG.BUCKET_NAME})",
    )
    parser.add_argument(
        "--prefix",
        "-x",
        dest="prefix",
        default=CONFIG.DEFAULT_S3_PREFIX,
        help=f"S3 prefix/folder inside the bucket (default: {CONFIG.DEFAULT_S3_PREFIX})",
    )
    parser.add_argument(
        "--region",
        "-r",
        dest="region",
        default=CONFIG.REGION,
        help=f"AWS region (default: {CONFIG.REGION})",
    )
    parser.add_argument(
        "--profile",
        dest="profile",
        default=CONFIG.AWS_PROFILE,
        help=f"AWS profile name (default: {CONFIG.AWS_PROFILE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be uploaded without actually uploading.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logger.info("=== S3 Uploader starting ===")
    logger.info("Local path : %s", args.local_path)
    logger.info("Bucket     : %s", args.bucket)
    logger.info("Prefix     : %s", args.prefix)
    logger.info("Region     : %s", args.region)
    logger.info("Profile    : %s", args.profile or "<default>")
    logger.info("Dry run    : %s", args.dry_run)

    try:
        session = create_boto3_session(args.profile, args.region)
        upload_path(
            session=session,
            bucket=args.bucket,
            local_path=args.local_path,
            base_prefix=args.prefix,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        logger.error("Uploader finished with error: %s", exc)
        return 1

    logger.info("=== S3 Uploader finished successfully ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
