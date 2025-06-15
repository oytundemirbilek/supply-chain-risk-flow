"""Common functions for pytest modules."""

import logging
from typing import Generator

import pytest


@pytest.fixture
def logger_test() -> Generator[logging.Logger, None, None]:
    """Yield a logger for all tests."""
    yield logging.getLogger(__name__)
    logging.shutdown()
