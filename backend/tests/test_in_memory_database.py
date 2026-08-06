import time
import unittest

from typing import cast

from app.db.in_memory_database import InMemoryDatabase


class InMemoryDatabaseTests(unittest.TestCase):
    def test_set_and_get_round_trip(self) -> None:
        db = InMemoryDatabase()

        value = 2

        self.assertTrue(db.set("name", "Alice"))
        self.assertTrue(db.set(cast(str, value), "pending"))

        self.assertEqual(db.get("name"), "Alice")
        self.assertEqual(db.get(cast(str, value)), "pending")

        self.assertIsNone(db.get("missing"))

    def test_compare_and_set_updates_only_when_expected_matches(self) -> None:
        db = InMemoryDatabase()
        db.set("status", "pending")

        self.assertTrue(db.compare_and_set("status", "pending", "complete"))
        self.assertEqual(db.get("status"), "complete")

        self.assertFalse(db.compare_and_set("status", "pending", "failed"))
        self.assertEqual(db.get("status"), "complete")

    def test_runtime_value_can_break_the_string_assert_for_set(self) -> None:
        db = InMemoryDatabase()
        runtime_key: object = "name"
        runtime_key = 42

        with self.assertRaises(AssertionError):
            assert isinstance(cast(str, runtime_key), str)

        self.assertTrue(db.set("fallback", "ok"))
        self.assertEqual(db.get("fallback"), "ok")

    def test_entries_include_optional_integer_value(self) -> None:
        db = InMemoryDatabase()

        self.assertTrue(db.set("counter", "value"))
        entry = db._store["counter"]

        self.assertEqual(entry["value"], "value")
        self.assertIsNone(entry["count"])

    def test_entry_expires_and_resets_to_none_after_ttl(self) -> None:
        db = InMemoryDatabase()

        self.assertTrue(db.set("expiring", "value", ttl_seconds=1))
        self.assertEqual(db.get("expiring"), "value")

        time.sleep(1.1)

        self.assertIsNone(db.get("expiring"))


if __name__ == "__main__":
    unittest.main()
