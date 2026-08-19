   #!/usr/bin/env python3
"""Extract Baloto results from the public results page and normalize them for RAG ingestion.

This script intentionally stays outside the app codebase and uses Playwright to:
1. Open the Baloto results page.
2. Find the table with the exact class list: "table table-hover gotham-book"
3. Read every row and normalize the data into a list of dictionaries.
4. Click the exact "Siguiente" link with class "btn btn-yellow w-100" and href "?page=2"
5. Repeat until that exact link is no longer present.

The output is a JSON list with rows normalized as:
[
  {
    "date": "15 de enero de 2024",
    "numbers": [12, 34, 56, 78, 90],
    "raw_numbers": "12 34 56 78 90"
  }
]
"""

from __future__ import annotations

from datetime import datetime
import json
import re
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

BASE_URL = "https://baloto.com/resultados"
TARGET_TABLE_CLASS = "table table-hover gotham-book"
PAGE_TOTAL_STRONG_SELECTOR = 'strong.current.ms-3.me-3'


def normalize_numbers(raw_numbers: str) -> list[int]:
    """Return only integer values found in the raw numeric cell."""
    numbers = re.findall(r"\d+", raw_numbers)
    return [int(number) for number in numbers]


def find_page_total(page) -> int | None:
    """Read the strong element like: 'Página 1 de 12' and return the total page count."""
    target = page.locator(PAGE_TOTAL_STRONG_SELECTOR)
    if target.count() == 0:
        print("[DEBUG] No page-total strong element found")
        return None

    text = target.first.inner_text().strip()
    print(f"[DEBUG] Page-total text: {text!r}")

    match = re.search(r"de\s+(\d+)", text, flags=re.IGNORECASE)
    if not match:
        print("[DEBUG] Could not parse total page count from strong element")
        return None

    total_pages = int(match.group(1))
    print(f"[DEBUG] Total pages detected: {total_pages}")
    return total_pages


def find_exact_table(page):
    """Return the table whose class attribute exactly matches the target class string."""
    tables = page.locator("table")
    table_count = tables.count()
    print(f"[DEBUG] Found {table_count} tables on page")
    for index in range(table_count):
        table = tables.nth(index)
        class_name = table.get_attribute("class") or ""
        print(f"[DEBUG] table[{index}] class={class_name!r}")
        if class_name == TARGET_TABLE_CLASS:
            print(f"[DEBUG] Exact table match found at table[{index}]")
            return table
    print("[DEBUG] No exact table match found")
    return None


def extract_rows_from_table(page) -> list[dict[str, Any]]:
    """Extract rows from the currently loaded Baloto page.

    The table rows have this structure:
    1) icon cell (ignored)
    2) date cell
    3) numbers cell
    4) detail link cell (ignored)
    """
    months = {
        "enero": 1,
        "febrero": 2,
        "marzo": 3,
        "abril": 4,
        "mayo": 5,
        "junio": 6,
        "julio": 7,
        "agosto": 8,
        "septiembre": 9,
        "octubre": 10,
        "noviembre": 11,
        "diciembre": 12,
    }
    table = find_exact_table(page)
    if table is None:
        return []
    rows = table.locator("tr")
    total_rows = rows.count()-1 # Exclude the header row
    print(f"[DEBUG] Table found with {total_rows} rows")

    extracted: list[dict[str, Any]] = []

    for index in range(total_rows):
        row = rows.nth(index)
        cells = row.locator("td")
        cell_count = cells.count()

        if cell_count < 4:
            continue

        date_value = cells.nth(1).inner_text().strip()
        number_value = cells.nth(2).inner_text().strip()

        day, _, month, _, year = date_value.lower().split()
        date = datetime(
            int(year),
            months[month],
            int(day),
        )

        if not date_value or not number_value:
            continue

        extracted.append(
            {
                "date": date.date().isoformat(),
                "numbers": normalize_numbers(number_value),
                "raw_numbers": number_value,
            }
        )

    return extracted


def save_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    output_path = Path(__file__).resolve().parent / "output" / "baloto_results.json"
    all_rows: list[dict[str, Any]] = []

    print("[DEBUG] Starting Baloto extraction script")
    with sync_playwright() as p:
        print("[DEBUG] Launching browser in headless mode")
        browser = p.chromium.launch(headless=True)
        print("[DEBUG] Browser launched")
        page = browser.new_page()
        print(f"[DEBUG] Going to {BASE_URL}")
        page.goto(BASE_URL, wait_until="networkidle")
        print(f"[DEBUG] Page loaded: {page.url}")
        print(f"[DEBUG] Page title: {page.title()}")

        total_pages = find_page_total(page)
        if total_pages is None:
            print("[DEBUG] No total page count found; stopping before extraction")
            browser.close()
            return

        for page_number in range(1, total_pages + 1):
            target_url = f"{BASE_URL}?page={page_number}"
            print(f"[DEBUG] Processing page {page_number} -> {target_url}")
            page.goto(target_url, wait_until="networkidle")

            print("[DEBUG] Extracting rows from current page")
            page_rows = extract_rows_from_table(page)
            print(f"[DEBUG] Extracted {len(page_rows)} rows from this page")
            all_rows.extend(page_rows)

        print(f"[DEBUG] Closing browser after processing {total_pages} page(s)")
        browser.close()

    print(f"[DEBUG] Saving {len(all_rows)} total rows to {output_path}")
    save_results(all_rows, output_path)

    print(json.dumps({
        "rows_extracted": len(all_rows),
        "output_file": str(output_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
