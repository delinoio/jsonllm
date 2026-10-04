"""Serializable failure details without stack-frame local variables."""

import traceback


def exception_details(exc):
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(
            traceback.TracebackException.from_exception(exc, capture_locals=False).format()
        ),
    }
