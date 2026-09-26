"""Destructively clear the clinical_chunks pgvector table after explicit opt-in."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv


TABLE_NAME = "clinical_chunks"


async def clear_table(database_url: str) -> int:
    import asyncpg

    connection_url = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    connection = await asyncpg.connect(connection_url, timeout=15, command_timeout=60)
    try:
        relation = await connection.fetchval("SELECT to_regclass($1)", TABLE_NAME)
        if relation is None:
            raise RuntimeError(f"Table {TABLE_NAME!r} does not exist; nothing was cleared.")
        previous_count = await connection.fetchval(f"SELECT count(*) FROM {TABLE_NAME}")
        await connection.execute(f"TRUNCATE TABLE {TABLE_NAME}")
        return previous_count
    finally:
        await connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Clear all passages and vectors from clinical_chunks.")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required confirmation; permanently deletes every row in clinical_chunks.",
    )
    args = parser.parse_args()
    if not args.confirm:
        parser.error("refusing to clear data without --confirm")

    load_dotenv(override=False)
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        print("DATABASE_URL is required in the environment or local .env file.", file=sys.stderr)
        return 2
    try:
        count = asyncio.run(clear_table(database_url))
    except Exception as exc:
        print(f"Could not clear {TABLE_NAME}: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(f"Cleared {count} row(s) from {TABLE_NAME}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
