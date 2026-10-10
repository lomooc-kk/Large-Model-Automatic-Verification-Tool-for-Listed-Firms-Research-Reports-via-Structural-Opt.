"""Cooperative cancellation shared by Pi business tools and paid transports."""
from contextlib import contextmanager
from contextvars import ContextVar
import time

_scope = ContextVar("yjcheck_execution_scope", default=None)


def check_active():
    scope = _scope.get()
    if scope and (scope[0].is_set() or time.monotonic() >= scope[1]):
        raise TimeoutError("agent_tool_cancelled")


def remaining_timeout(default):
    check_active()
    scope = _scope.get()
    return min(default, max(0.001, scope[1] - time.monotonic())) if scope else default


@contextmanager
def execution_scope(cancelled, deadline):
    token = _scope.set((cancelled, deadline))
    try:
        check_active()
        yield
    finally:
        _scope.reset(token)
