"""One stable stdout/stderr lifetime per node; stream errors fail the node."""
from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import sys
import traceback
from pathlib import Path


class FailFastStreamHandler(logging.StreamHandler):
    def handleError(self, record):
        error = sys.exc_info()[1]
        raise RuntimeError('Node logging output failed; scientific completion prohibited') from error


@contextlib.contextmanager
def bound_ultralytics_logger():
    from ultralytics.utils import LOGGER
    if getattr(sys.stdout, 'closed', False) or getattr(sys.stderr, 'closed', False):
        raise RuntimeError('Node stdout/stderr already closed')
    # Remove stale/duplicate stream handlers, preserve file and other sinks.
    detached = [h for h in LOGGER.handlers if isinstance(h, logging.StreamHandler)
                and not isinstance(h, logging.FileHandler)]
    for handler in detached:
        LOGGER.removeHandler(handler)
    handler = FailFastStreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    handler.setLevel(LOGGER.level)
    LOGGER.addHandler(handler)
    try:
        LOGGER.info('NODE_LOGGING_BOUND pid=%s', os.getpid())
        yield LOGGER
    except BaseException:
        # Direct traceback goes to the same live child descriptor, not a closed prior sink.
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        try:
            handler.flush()
        finally:
            LOGGER.removeHandler(handler)
            handler.close()
            for previous in detached:
                stream = getattr(previous, 'stream', None)
                if stream is not None and not getattr(stream, 'closed', False):
                    LOGGER.addHandler(previous)
                else:
                    previous.close()


def run_logged_process(argv, log_path, *, env=None):
    """Keep the parent's file open until child and its joined workers terminate."""
    path = Path(log_path)
    with path.open('x', encoding='utf-8') as stream:
        result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                                env=env, check=False)
        stream.flush()
        os.fsync(stream.fileno())
    if result.returncode:
        raise RuntimeError(f'Node child failed: exit={result.returncode}; full log={path}')
    return path

