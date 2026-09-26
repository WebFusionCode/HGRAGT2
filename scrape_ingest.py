"""Fetch authoritative clinical pages and index them in PostgreSQL/pgvector.

Run ``python scrape_ingest.py --dry-run`` to inspect extraction and chunking
without embeddings or database writes. A normal run requires DATABASE_URL and
the credentials for the LiteLLM embedding model selected in the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


TARGET_URLS = (
    "https://www.cdc.gov/heads-up/guidelines/recovery-from-concussion.html",
    "https://www.cdc.gov/traumatic-brain-injury/hcp/data-research/index.html",
    "https://www.aafp.org/afp/2019/0401/p426",
    "https://www.ncbi.nlm.nih.gov/books/NBK538149",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC5112330",
    "https://my.clevelandclinic.org/health/diseases/21553-achilles-tendinitis",
    "https://www.ncbi.nlm.nih.gov/books/NBK537017",
)

ALLOWED_HOSTS = {
    "cdc.gov", "www.cdc.gov", "aafp.org", "www.aafp.org",
    "ncbi.nlm.nih.gov", "www.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov",
    "my.clevelandclinic.org",
}
TABLE_NAME = "clinical_chunks"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 384
DEFAULT_BATCH_SIZE = 64
MAX_HTML_BYTES = 15 * 1024 * 1024
REQUEST_TIMEOUT = (8, 30)
MAX_RETRIES = 3
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
BLOCKED_PAGE_MARKERS = ("access denied", "forbidden", "javascript")
FALLBACK_TEXT = (
    "Concussion Management: Initial rest for 24-48 hours is recommended. After this period, patients should begin a gradual return to activity, staying below their symptom-exacerbation threshold. Limit screen time and physical activities in the first 1 to 2 days. Recovery timelines vary; 85% to 90% of adults recover within two weeks, whereas children typically take one to three months. Neuropsychological tests help identify cognitive deficits but are not well validated for initial diagnosis. The American College of Emergency Physicians updated their clinical policy for adult mTBI in 2023."
)
_LOGGER = logging.getLogger("scrape_ingest")


@dataclass(frozen=True)
class SourcePolicy:
    allowed_role: str
    allowed_roles: tuple[str, ...]


@dataclass
class ScrapedDocument:
    source_url: str
    title: str
    text: str
    host: str
    fetched_at: str
    http_last_modified: str | None
    published_date: str | None
    content_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def make_session() -> requests.Session:
    retry = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        status=MAX_RETRIES,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
    })
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session.mount("https://", adapter)
    return session


def _validate_public_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    if (
        parsed.scheme != "https"
        or host not in ALLOWED_HOSTS
        or parsed.port not in (None, 443)
        or parsed.username
        or parsed.password
    ):
        raise ValueError(f"URL is outside the HTTPS source allowlist: {url}")
    return host


def policy_for_url(url: str) -> SourcePolicy:
    """Assign least-privilege roles from the source class, not user input."""
    host = _validate_public_url(url)
    path = urlparse(url).path.lower()
    if host.endswith("cdc.gov") and "/heads-up/" in path:
        return SourcePolicy(allowed_role="patient", allowed_roles=("patient", "clinician"))
    if host.endswith("clevelandclinic.org"):
        return SourcePolicy(allowed_role="patient", allowed_roles=("patient", "clinician"))
    return SourcePolicy(allowed_role="clinician", allowed_roles=("clinician",))


def _read_limited_response(response: requests.Response, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    for block in response.iter_content(chunk_size=64 * 1024):
        if not block:
            continue
        total += len(block)
        if total > limit:
            raise ValueError(f"HTML response exceeded the {limit // (1024 * 1024)} MB safety limit")
        chunks.append(block)
    return b"".join(chunks)


def _robots_allows(session: requests.Session, url: str, cache: dict[str, RobotFileParser]) -> bool:
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    parser = cache.get(origin)
    if parser is None:
        robots_url = f"{origin}/robots.txt"
        response = session.get(robots_url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        try:
            _validate_public_url(response.url)
            if response.status_code in (404, 410):
                lines: list[str] = []
            else:
                response.raise_for_status()
                lines = response.text.splitlines()
        finally:
            response.close()
        parser = RobotFileParser(robots_url)
        parser.parse(lines)
        cache[origin] = parser
    return parser.can_fetch(USER_AGENT, url)


def scrape_url(
    session: requests.Session,
    url: str,
    robots_cache: dict[str, RobotFileParser] | None = None,
) -> ScrapedDocument:
    """Fetch and extract a page, retaining tables and its canonical URL."""
    _validate_public_url(url)
    if not _robots_allows(session, url, robots_cache if robots_cache is not None else {}):
        raise ValueError("Scraping is disallowed for this path by the site's robots.txt")
    response = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True, stream=True)
    try:
        _validate_public_url(response.url)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and "html" not in content_type and "xhtml" not in content_type:
            raise ValueError(f"Expected an HTML page, received {content_type}")
        html = _read_limited_response(response, MAX_HTML_BYTES)
        final_url = response.url
        host = _validate_public_url(final_url)
        fetched_at = utc_now()
        last_modified = response.headers.get("Last-Modified")
    finally:
        response.close()

    import trafilatura

    extracted = trafilatura.extract_with_metadata(
        html,
        url=final_url,
        include_comments=False,
        include_tables=True,
        include_formatting=True,
        favor_precision=True,
        deduplicate=True,
    )
    if extracted is None:
        raise ValueError("Trafilatura could not identify article content")

    text = re.sub(r"\n{3,}", "\n\n", (extracted.text or "").strip())
    if text.startswith("---\n"):
        metadata_end = text.find("\n---\n", 4)
        if metadata_end >= 0:
            text = text[metadata_end + 5:].lstrip()

    title = (getattr(extracted, "title", None) or "").strip() or host
    published_date = getattr(extracted, "date", None)
    if published_date:
        published_date = str(published_date)
    return ScrapedDocument(
        source_url=final_url,
        title=title,
        text=text,
        host=host,
        fetched_at=fetched_at,
        http_last_modified=last_modified,
        published_date=published_date,
        content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def validate_extracted_text(text: str) -> tuple[bool, str | None]:
    cleaned = text.strip()
    if len(cleaned) < 200:
        return False, f"only {len(cleaned)} characters extracted (minimum is 200)"
    lowered = cleaned.lower()
    for marker in BLOCKED_PAGE_MARKERS:
        if marker in lowered:
            return False, f"extracted content contains blocked-page marker {marker!r}"
    return True, None


def split_document(text: str) -> list[str]:
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError("Text chunking requires tiktoken; install requirements.txt") from exc

    encoding = tiktoken.get_encoding("cl100k_base")
    separators = ("\n\n", "\n", ". ", "; ", " ")

    def token_count(value: str) -> int:
        return len(encoding.encode(value, disallowed_special=()))

    def split_recursively(value: str, separator_index: int = 0) -> list[str]:
        if token_count(value) <= 500:
            return [value]

        for index in range(separator_index, len(separators)):
            separator = separators[index]
            if separator not in value:
                continue
            raw_parts = value.split(separator)
            parts = [part + (separator if part_index < len(raw_parts) - 1 else "")
                     for part_index, part in enumerate(raw_parts)]
            if len(parts) == 1:
                continue
            result: list[str] = []
            for part in parts:
                if not part:
                    continue
                if token_count(part) > 500:
                    result.extend(split_recursively(part, index + 1))
                else:
                    result.append(part)
            return result

        token_ids = encoding.encode(value, disallowed_special=())
        stride = 450
        return [encoding.decode(token_ids[start:start + stride]) for start in range(0, len(token_ids), stride)]

    units = split_recursively(re.sub(r"\n{3,}", "\n\n", text).strip())
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = current + unit
        if not current or token_count(candidate) <= 500:
            current = candidate
            continue

        if current.strip():
            chunks.append(current.strip())
        overlap = encoding.encode(current, disallowed_special=())[-50:]
        while overlap and token_count(encoding.decode(overlap) + unit) > 500:
            overlap = overlap[1:]
        current = encoding.decode(overlap) + unit
    if current.strip():
        chunks.append(current.strip())
    return chunks


def make_chunk_records(
    document: ScrapedDocument,
    policy: SourcePolicy | None = None,
    *,
    status: str = "active",
    source_type: str = "authoritative_web_guideline",
) -> list[dict[str, Any]]:
    policy = policy or policy_for_url(document.source_url)
    pieces = split_document(document.text)
    stable_source = document.source_url or f"demo-fallback:{document.content_sha256}"
    records: list[dict[str, Any]] = []
    for index, piece in enumerate(pieces, start=1):
        chunk_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{stable_source}#{index}")
        records.append({
            "id": chunk_id,
            "content": piece,
            "metadata": {
                "doc_id": str(uuid.uuid5(uuid.NAMESPACE_URL, stable_source)),
                "source_url": document.source_url,
                "citation_url": document.source_url or None,
                "source_host": document.host,
                "title": document.title,
                "published_date": document.published_date,
                "http_last_modified": document.http_last_modified,
                "fetched_at": document.fetched_at,
                "content_sha256": document.content_sha256,
                "chunk_number": index,
                "chunk_count": len(pieces),
                "allowed_role": policy.allowed_role,
                "allowed_roles": list(policy.allowed_roles),
                "status": status,
                "source_type": source_type,
                "demo_only": status == "demo_fallback",
                "verification_status": "unverified" if status == "demo_fallback" else "source_extracted",
            },
        })
    return records


def make_fallback_document() -> tuple[ScrapedDocument, list[dict[str, Any]]]:
    document = ScrapedDocument(
        source_url="",
        title="DEMO ONLY - Unverified concussion fallback; not an authoritative source",
        text=FALLBACK_TEXT,
        host="demo-fallback",
        fetched_at=utc_now(),
        http_last_modified=None,
        published_date=None,
        content_sha256=hashlib.sha256(FALLBACK_TEXT.encode("utf-8")).hexdigest(),
    )
    records = make_chunk_records(
        document,
        SourcePolicy(allowed_role="clinician", allowed_roles=("clinician",)),
        status="demo_fallback",
        source_type="unverified_demo_fallback",
    )
    return document, records


def _is_transient_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    markers = ("timeout", "connection", "temporar", "rate limit", "ratelimit", "429", "502", "503", "504", "server closed")
    return any(marker in name or marker in message for marker in markers)


async def embed_chunks(chunks: list[dict[str, Any]], model: str, dimensions: int, batch_size: int) -> None:
    import litellm

    def field(value: Any, name: str) -> Any:
        return value[name] if isinstance(value, dict) else getattr(value, name)

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [chunk["content"] for chunk in batch]
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await litellm.aembedding(
                    model=model,
                    input=texts,
                    dimensions=dimensions,
                )
                items = sorted(response.data, key=lambda item: field(item, "index"))
                if len(items) != len(batch):
                    raise ValueError("Embedding provider returned a different number of vectors than inputs")
                for chunk, item in zip(batch, items):
                    vector = field(item, "embedding")
                    if len(vector) != dimensions:
                        raise ValueError(
                            f"Embedding dimension mismatch: expected {dimensions}, received {len(vector)}"
                        )
                    chunk["embedding"] = vector
                break
            except Exception as exc:
                if attempt >= MAX_RETRIES or not _is_transient_error(exc):
                    raise RuntimeError(f"Embedding batch {start // batch_size + 1} failed: {type(exc).__name__}") from exc
                delay = min(2 ** attempt, 8)
                _LOGGER.warning("Embedding request timed out or was throttled; retrying in %ss", delay)
                await asyncio.sleep(delay)


async def _create_schema(conn: Any, dimensions: int) -> None:
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id UUID PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector({dimensions}) NOT NULL,
            chunk_metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
        )
    """)

    row = await conn.fetchrow("""
        SELECT format_type(attribute.atttypid, attribute.atttypmod) AS embedding_type
        FROM pg_attribute AS attribute
        WHERE attribute.attrelid = $1::regclass
          AND attribute.attname = 'embedding'
          AND NOT attribute.attisdropped
    """, TABLE_NAME)
    if row is None:
        raise RuntimeError(f"{TABLE_NAME} exists without an embedding column")
    match = re.fullmatch(r"vector\((\d+)\)", row["embedding_type"])
    if not match or int(match.group(1)) != dimensions:
        raise RuntimeError(
            f"{TABLE_NAME}.embedding is {row['embedding_type']}; configured dimensions are {dimensions}. "
            "Set EMBEDDING_DIMENSIONS to the table's existing dimension or migrate the table deliberately."
        )

    await conn.execute(
        f"CREATE INDEX IF NOT EXISTS clinical_chunks_embedding_hnsw_idx "
        f"ON {TABLE_NAME} USING hnsw (embedding vector_cosine_ops)"
    )
    await conn.execute(
        f"CREATE INDEX IF NOT EXISTS clinical_chunks_metadata_gin_idx "
        f"ON {TABLE_NAME} USING gin (chunk_metadata)"
    )


async def _insert_chunks_once(
    chunks: list[dict[str, Any]], database_url: str, dimensions: int, batch_size: int
) -> int:
    import asyncpg
    from pgvector import Vector
    from pgvector.asyncpg import register_vector

    connection_url = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    conn = await asyncpg.connect(connection_url, timeout=15, command_timeout=60)
    try:
        await _create_schema(conn, dimensions)
        await register_vector(conn)
        rows = [
            (chunk["id"], chunk["content"], Vector(chunk["embedding"]), json.dumps(chunk["metadata"], ensure_ascii=True))
            for chunk in chunks
        ]
        source_urls = sorted({chunk["metadata"]["source_url"] for chunk in chunks})

        # Replace only successfully fetched sources; old rows survive any failed transaction.
        async with conn.transaction():
            if any(source_url for source_url in source_urls):
                await conn.execute(
                    f"DELETE FROM {TABLE_NAME} WHERE chunk_metadata->>'source_type' = $1",
                    "unverified_demo_fallback",
                    timeout=30,
                )
            for source_url in source_urls:
                await conn.execute(
                    f"DELETE FROM {TABLE_NAME} WHERE chunk_metadata->>'source_url' = $1",
                    source_url,
                    timeout=30,
                )
            query = f"""
                INSERT INTO {TABLE_NAME} (id, content, embedding, chunk_metadata)
                VALUES ($1, $2, $3, $4::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    chunk_metadata = EXCLUDED.chunk_metadata
            """
            for start in range(0, len(rows), batch_size):
                await conn.executemany(query, rows[start:start + batch_size], timeout=60)
        return len(rows)
    finally:
        await conn.close()


async def insert_chunks(
    chunks: list[dict[str, Any]], database_url: str, dimensions: int, batch_size: int
) -> int:
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await _insert_chunks_once(chunks, database_url, dimensions, batch_size)
        except Exception as exc:
            if attempt >= MAX_RETRIES or not _is_transient_error(exc):
                raise
            delay = min(2 ** attempt, 8)
            _LOGGER.warning("Database connection failed; retrying transaction in %ss", delay)
            await asyncio.sleep(delay)
    raise RuntimeError("Database insertion exhausted retries")


def _console() -> Any:
    try:
        from rich.console import Console
        return Console()
    except ImportError:
        return None


def _print_results(results: list[dict[str, Any]]) -> None:
    console = _console()
    if console is None:
        for result in results:
            print(f"{result['status']:>8} | {result['chunks']:>3} chunks | {result['url']}")
            if result.get("error"):
                print(f"           {result['error']}")
        return

    from rich.table import Table

    table = Table(title="Clinical source ingestion", header_style="bold cyan", show_lines=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Source", max_width=70)
    table.add_column("Title", max_width=38)
    table.add_column("Chunks", justify="right")
    for result in results:
        if result["status"] in {"ready", "inserted", "dry-run"}:
            status_style = "green"
        elif result["status"] == "demo-fallback":
            status_style = "yellow"
        else:
            status_style = "red"
        source = result["url"]
        if result.get("error"):
            source += f"\n[red]{result['error']}[/red]"
        table.add_row(
            f"[{status_style}]{result['status']}[/{status_style}]",
            source,
            result.get("title", "-"),
            str(result["chunks"]),
        )
    console.print(table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape trusted clinical pages and ingest them into pgvector.")
    parser.add_argument("--dry-run", action="store_true", help="Scrape and chunk only; skip embeddings and database writes.")
    parser.add_argument("--url", action="append", dest="urls", help="Additional HTTPS URL from an allowlisted host; repeatable.")
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("INGEST_BATCH_SIZE", DEFAULT_BATCH_SIZE)))
    return parser


async def run(args: argparse.Namespace) -> int:
    urls = list(dict.fromkeys([*TARGET_URLS, *(args.urls or [])]))
    if not urls:
        _LOGGER.error("No source URLs were configured.")
        return 2
    if args.batch_size < 1:
        _LOGGER.error("--batch-size must be positive.")
        return 2

    documents: list[ScrapedDocument] = []
    chunks: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    session = make_session()
    robots_cache: dict[str, RobotFileParser] = {}
    try:
        try:
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
            progress = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn())
        except ImportError:
            progress = None

        if progress:
            progress.start()
            task_id = progress.add_task("Scraping clinical sources", total=len(urls))
        else:
            task_id = None

        for url in urls:
            if progress:
                progress.update(task_id, description=f"Scraping {urlparse(url).netloc}{urlparse(url).path[:34]}")
            else:
                _LOGGER.info("Scraping %s", url)
            try:
                document = scrape_url(session, url, robots_cache)
                valid, reason = validate_extracted_text(document.text)
                if not valid:
                    message = reason or "validation failed"
                    print(f"[WARNING] Skipping {url}: {message}.")
                    results.append({"url": url, "title": "-", "chunks": 0, "status": "failed", "error": message})
                    continue

                preview = document.text[:500]
                print(f"\n[EXTRACT PREVIEW] {document.source_url}\n{preview}\n")
                source_chunks = make_chunk_records(document)
                documents.append(document)
                chunks.extend(source_chunks)
                results.append({"url": document.source_url, "title": document.title, "chunks": len(source_chunks), "status": "ready"})
            except Exception as exc:
                message = str(exc) if isinstance(exc, (ValueError, requests.RequestException)) else type(exc).__name__
                print(f"[WARNING] Skipping {url}: scraping/extraction failed ({message}).")
                results.append({"url": url, "title": "-", "chunks": 0, "status": "failed", "error": message})
            finally:
                if progress:
                    progress.advance(task_id)

        if progress:
            progress.stop()
    finally:
        session.close()

    using_fallback = not documents
    if using_fallback:
        _LOGGER.warning("All source pages failed validation; using the unverified demo-only fallback text.")
        fallback_document, fallback_chunks = make_fallback_document()
        chunks.extend(fallback_chunks)
        results.append({
            "url": "DEMO ONLY - no citation URL",
            "title": fallback_document.title,
            "chunks": len(fallback_chunks),
            "status": "demo-fallback",
        })

    if args.dry_run:
        for result in results:
            if result["status"] == "ready":
                result["status"] = "dry-run"
        _print_results(results)
        _LOGGER.info("Dry run complete: %s source(s), %s passage(s). No embeddings or database writes performed.", len(documents), len(chunks))
        return 1 if any(item["status"] == "failed" for item in results) and not using_fallback else 0

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        _print_results(results)
        _LOGGER.error("DATABASE_URL is required. Set it in the environment or a local .env file.")
        return 2

    model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
    try:
        dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", str(DEFAULT_EMBEDDING_DIMENSIONS)))
    except ValueError:
        _LOGGER.error("EMBEDDING_DIMENSIONS must be a positive integer.")
        return 2
    if dimensions < 1:
        _LOGGER.error("EMBEDDING_DIMENSIONS must be a positive integer.")
        return 2

    try:
        _LOGGER.info("Embedding %s passages with %s (%s dimensions)", len(chunks), model, dimensions)
        await embed_chunks(chunks, model, dimensions, args.batch_size)
        inserted = await insert_chunks(chunks, database_url, dimensions, args.batch_size)
    except Exception as exc:
        _LOGGER.error("Ingestion stopped without replacing indexed sources (%s)", type(exc).__name__)
        _print_results(results)
        return 1

    for result in results:
        if result["status"] == "ready":
            result["status"] = "inserted"
    _print_results(results)
    _LOGGER.info("Database transaction committed: %s passage(s) indexed.", inserted)
    failed = any(item["status"] == "failed" for item in results) and not using_fallback
    return 1 if failed else 0


def main() -> int:
    load_dotenv(override=False)
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        _LOGGER.warning("Ingestion interrupted; no uncommitted database changes were kept.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
