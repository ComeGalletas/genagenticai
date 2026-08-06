import abc
import time
from typing import Any


class InMemoryDatabaseInterface(abc.ABC):
    @abc.abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> bool:
        """Store a value and return True when the write succeeds."""

    @abc.abstractmethod
    def get(self, key: str) -> Any:
        """Return the stored value for a key, or None if it does not exist."""

    @abc.abstractmethod
    def compare_and_set(
        self,
        key: str,
        expected_value: Any,
        new_value: Any,
    ) -> bool:
        """Replace the value only when the current value matches expected_value."""


class InMemoryDatabase(InMemoryDatabaseInterface):
    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> bool:
        expires_at = None
        if ttl_seconds is not None:
            expires_at = time.time() + ttl_seconds

        self._store[key] = {"value": value, "count": expires_at}
        return True

    def get(self, key: str) -> Any:
        entry = self._store.get(key)
        if entry is None:
            return None

        expires_at = entry["count"]
        if expires_at is not None and time.time() >= expires_at:
            self._store[key] = {"value": None, "count": None}
            return None

        return entry["value"]

    def compare_and_set(
        self,
        key: str,
        expected_value: Any,
        new_value: Any,
    ) -> bool:
        entry = self._store.get(key)
        if entry is None or entry["value"] != expected_value:
            return False

        entry["value"] = new_value
        return True
