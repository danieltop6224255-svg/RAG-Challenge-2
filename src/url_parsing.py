import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import trafilatura

logger = logging.getLogger(__name__)


class URLParser:
    """Parse URLs with trafilatura and export reports in pipeline-compatible JSON.

    Export schema matches processed reports consumed by text splitter/ingestion:
    {
      "metainfo": {
        "sha1_name": "...",
        "source_type": "url",
        "url": "..."
      },
      "content": {
        "chunks": null,
        "pages": [{"page": 1, "text": "..."}]
      }
    }
    """

    def __init__(
        self,
        user_agent: Optional[str] = None,
        crawl_delay: float = 0.5,
        include_tables: bool = True,
        output_format: str = "markdown",
        output_dir: Optional[Path] = None,
    ):
        self.user_agent = user_agent or "rag-challenge-url-parser/1.0"
        self.crawl_delay = crawl_delay
        self.include_tables = include_tables
        self.output_format = output_format if output_format in ("plain", "markdown") else "markdown"
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _extract_with_trafilatura(self, html: str) -> Optional[str]:
        if not html:
            return None
        return trafilatura.extract(
            html,
            include_comments=False,
            include_tables=self.include_tables,
            output_format=self.output_format,
        )

    def _fetch_with_trafilatura(self, url: str) -> Optional[str]:
        try:
            return trafilatura.fetch_url(url)
        except Exception as error:
            logger.debug("trafilatura.fetch_url error for %s: %s", url, error)
            return None

    def _safe_filename(self, url: str, suffix: str = ".json") -> str:
        parsed = urlparse(url)
        base = parsed.netloc + parsed.path
        if parsed.query:
            base += "?" + parsed.query
        name = re.sub(r"[^0-9A-Za-z]+", "_", base).strip("_")
        if not name:
            name = hashlib.sha1(url.encode("utf-8")).hexdigest()
        if len(name) > 200:
            name = name[:200]
        short_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
        return f"{name}_{short_hash}{suffix}"

    @staticmethod
    def _get_sha1_name(url: str) -> str:
        return hashlib.sha1(url.encode("utf-8")).hexdigest()

    def _build_output_payload(self, url: str, text: str) -> Dict:
        sha1_name = self._get_sha1_name(url)

        return {
            "metainfo": {
                "sha1_name": sha1_name,
                "source_type": "url",
                "url": url,
            },
            "content": {"chunks": None, "pages": [{"page": 1, "text": text}]},
        }

    def parse_urls(self, urls: List[Union[str, Dict[str, str]]]) -> None:
        """Parse URL list and write one JSON report per URL.

        `urls` accepts either:
        - list[str]
        - list[{"url": "..."}]
        """
        for index, item in enumerate(urls):
            if isinstance(item, str):
                url = item
            else:
                url = item.get("url", "")

            if not url:
                logger.warning("Skipping malformed URL record at index %d", index)
                continue

            logger.info("Parsing %d/%d: %s", index + 1, len(urls), url)
            time.sleep(self.crawl_delay)

            downloaded = self._fetch_with_trafilatura(url)
            extracted_text = self._extract_with_trafilatura(downloaded) if downloaded else None

            if not extracted_text:
                logger.warning("Failed to extract text from %s (skipping)", url)
                continue

            json_obj = self._build_output_payload(url, extracted_text)
            out_path = self.output_dir / self._safe_filename(url)

            with out_path.open("w", encoding="utf-8") as file:
                json.dump(json_obj, file, ensure_ascii=False, indent=2)
            logger.info("Saved parsed URL data to %s", out_path)
