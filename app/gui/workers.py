"""Threaded workers for heavy DSP actions."""

from __future__ import annotations

import traceback
from typing import Any, Callable

from PySide6 import QtCore


class WorkerSignals(QtCore.QObject):
    """Signals emitted by a background worker."""

    finished = QtCore.Signal(object)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(int)
    log = QtCore.Signal(str)


class Worker(QtCore.QRunnable):
    """Generic QRunnable wrapper around a callable."""

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result = self.fn(
                *self.args,
                progress_callback=self.signals.progress.emit,
                log_callback=self.signals.log.emit,
                **self.kwargs,
            )
        except Exception:
            self.signals.error.emit(traceback.format_exc())
            return
        self.signals.finished.emit(result)
