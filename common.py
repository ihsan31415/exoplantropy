"""Compatibility wrapper to keep pickled models referencing the legacy `common` module working."""
from scripts.common import *  # noqa: F401,F403
