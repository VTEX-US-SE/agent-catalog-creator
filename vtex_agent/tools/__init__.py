"""Tools for legacy site extraction and processing.

Imports are lazy so ``from vtex_agent.tools.image_manager import …`` does not
load crawler or Gemini dependencies unless the package ``__init__`` attributes
are accessed.
"""

from __future__ import annotations

__all__ = [
    "extract_sitemap_urls",
    "recursive_crawl_pdp_patterns",
    "crawl_categories",
    "build_session",
    "parse_category_tree_from_url",
    "extract_to_vtex_schema",
    "analyze_structure_from_sample",
    "extract_high_res_images",
    "prompt_manager_cli_main",
]


def __getattr__(name: str):
    if name in (
        "extract_sitemap_urls",
        "recursive_crawl_pdp_patterns",
        "crawl_categories",
        "build_session",
    ):
        from . import sitemap_crawler as _m

        return getattr(_m, name)
    if name == "parse_category_tree_from_url":
        from .url_parser import parse_category_tree_from_url

        return parse_category_tree_from_url
    if name in ("extract_to_vtex_schema", "analyze_structure_from_sample"):
        from . import gemini_mapper as _m

        return getattr(_m, name)
    if name == "extract_high_res_images":
        from .image_manager import extract_high_res_images

        return extract_high_res_images
    if name == "prompt_manager_cli_main":
        from .prompt_manager_cli import main as prompt_manager_cli_main

        return prompt_manager_cli_main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
