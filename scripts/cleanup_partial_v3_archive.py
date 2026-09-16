#!/usr/bin/env python3
"""Delete only v3 source files fully verified in the stopped partial archive."""

import os
import signal
import stat
import subprocess
import tarfile
from pathlib import PurePosixPath


ARCHIVE = "/e/project1/cameo/schmidt36/cameo_cg/local_work/archive/v3_stage3_meanforce_20260824.tar.zst"
SOURCE = "/e/project1/cameo/schmidt36/cameo_cg/local_work/v3_stage3_meanforce"
ROOT_NAME = "v3_stage3_meanforce"
CHUNK_SIZE = 8 * 1024 * 1024
ZSTD = "/e/project1/cameo/schmidt36/cameo_cg/local_work/archive/tools/zstd-aarch64"


class StopRequested(Exception):
    pass


def request_stop(signum, frame):
    raise StopRequested(f"received signal {signum}")


signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)

complete_members = 0
complete_regular = 0
complete_bytes = 0
deleted_files = 0
deleted_bytes = 0
missing_or_mismatched = 0
unsafe_members = 0
delete_failures = 0
empty_dirs_removed = 0
last_member = None
last_complete = None
stream_error = None
last_report = 0
proc = None

try:
    proc = subprocess.Popen(
        [ZSTD, "-d", "-T0", "-c", "--", ARCHIVE],
        stdout=subprocess.PIPE,
    )
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
        while True:
            try:
                member = tf.next()
            except Exception as exc:
                stream_error = f"{type(exc).__name__}: {exc}"
                break
            if member is None:
                break

            last_member = member.name
            rel = PurePosixPath(member.name)
            if (
                not rel.parts
                or rel.parts[0] != ROOT_NAME
                or rel.is_absolute()
                or ".." in rel.parts
            ):
                unsafe_members += 1
                stream_error = f"unsafe archive member: {member.name!r}"
                break

            if member.isfile():
                remaining = member.size
                try:
                    payload = tf.extractfile(member)
                    if payload is None:
                        raise RuntimeError("extractfile returned None")
                    while remaining:
                        data = payload.read(min(CHUNK_SIZE, remaining))
                        if not data:
                            break
                        remaining -= len(data)
                    payload.close()
                except Exception as exc:
                    stream_error = f"{type(exc).__name__}: {exc}"
                    break
                if remaining:
                    stream_error = (
                        f"incomplete payload for {member.name!r}; "
                        f"{remaining} bytes remain"
                    )
                    break

            complete_members += 1
            last_complete = member.name
            if not member.isfile():
                continue

            complete_regular += 1
            complete_bytes += member.size
            target = os.path.join(SOURCE, *rel.parts[1:])

            parent = SOURCE
            parent_safe = True
            for part in rel.parts[1:-1]:
                parent = os.path.join(parent, part)
                if os.path.islink(parent):
                    parent_safe = False
                    break

            if not parent_safe or os.path.islink(target):
                unsafe_members += 1
            else:
                try:
                    source_stat = os.lstat(target)
                    if stat.S_ISREG(source_stat.st_mode) and source_stat.st_size == member.size:
                        os.unlink(target)
                        deleted_files += 1
                        deleted_bytes += member.size
                    else:
                        missing_or_mismatched += 1
                except FileNotFoundError:
                    missing_or_mismatched += 1
                except OSError:
                    delete_failures += 1

            if complete_regular >= last_report + 50_000:
                last_report = complete_regular
                print(
                    f"progress complete_regular={complete_regular} "
                    f"complete_bytes={complete_bytes} deleted_files={deleted_files} "
                    f"deleted_bytes={deleted_bytes}",
                    flush=True,
                )
except StopRequested as exc:
    stream_error = str(exc)
except Exception as exc:
    stream_error = f"{type(exc).__name__}: {exc}"
finally:
    if proc is not None:
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

if stream_error is None and proc is not None and proc.returncode:
    stream_error = f"zstd exited with status {proc.returncode}"

if stream_error is None:
    for root, dirs, files in os.walk(SOURCE, topdown=False):
        if root == SOURCE:
            continue
        try:
            os.rmdir(root)
            empty_dirs_removed += 1
        except OSError:
            pass

print(f"stream_complete={stream_error is None}")
print(f"stream_error={stream_error!r}")
print(f"complete_members={complete_members}")
print(f"complete_regular_members={complete_regular}")
print(f"complete_payload_bytes={complete_bytes}")
print(f"deleted_files={deleted_files}")
print(f"deleted_bytes={deleted_bytes}")
print(f"missing_or_mismatched={missing_or_mismatched}")
print(f"unsafe_members={unsafe_members}")
print(f"delete_failures={delete_failures}")
print(f"empty_dirs_removed={empty_dirs_removed}")
print(f"last_complete_member={last_complete!r}")
print(f"last_seen_member={last_member!r}")
