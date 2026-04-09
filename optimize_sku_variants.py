#!/usr/bin/env python3
"""
Normalize SKU specifications so visual attributes stay constant per product
while non-visual 'switch' dimensions vary — enabling one shared image set per product.

Also assigns exactly four image URLs per product (views 1–4), named
``{ProductId}_{1..4}`` only (slugified); every SKU in the product gets the same list.

Run: python3 optimize_sku_variants.py
Rewrites: generated_data.json (same directory as this script)
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path


DATA_FILE = Path(__file__).resolve().parent / "generated_data.json"

# (pattern_tuple): lock = constant across all SKUs of the product; vary = differentiators
PATTERN_RULES: dict[tuple[str, ...], dict[str, list[str]]] = {
    ("Color", "Material", "Size"): {"lock": ["Color", "Material"], "vary": ["Size"]},
    ("Color", "Dimensions", "Material"): {"lock": ["Color", "Material"], "vary": ["Dimensions"]},
    ("Color", "Connectivity", "Warranty"): {"lock": ["Color"], "vary": ["Connectivity", "Warranty"]},
    ("Color", "Material", "Size", "Use Level"): {
        "lock": ["Color", "Material"],
        "vary": ["Size", "Use Level"],
    },
    ("Life Stage", "Pet Type", "Weight"): {"lock": ["Life Stage", "Pet Type"], "vary": ["Weight"]},
    ("Format", "Strength", "Use"): {"lock": ["Format", "Use"], "vary": ["Strength"]},
    ("Age Range", "Material", "Safety"): {"lock": ["Material", "Safety"], "vary": ["Age Range"]},
    ("Color", "Format", "Pack Size"): {"lock": ["Color", "Format"], "vary": ["Pack Size"]},
    ("Compatibility", "Material", "Voltage"): {"lock": ["Material"], "vary": ["Compatibility", "Voltage"]},
    ("Material", "Power Source", "Use"): {"lock": ["Material"], "vary": ["Power Source", "Use"]},
    ("Form", "Skin/Hair Type", "Volume"): {"lock": ["Form", "Skin/Hair Type"], "vary": ["Volume"]},
    ("Diet", "Pack Size", "Storage"): {"lock": ["Diet", "Storage"], "vary": ["Pack Size"]},
}

DEFAULT_SIZE_POOL = ["S", "M", "L", "XL", "One Size", "XXL"]

# Shared gallery size (matches generate_catalog_images.py NUM_VIEWS_TARGET).
NUM_IMAGES_PER_PRODUCT = 4

_DEFAULT_RAW_BASE = (
    "https://raw.githubusercontent.com/VTEX-US-SE/poc-vtex-day-2026/main/images"
)


def _slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "item"


def _string_urls(images: object) -> list[str]:
    if not isinstance(images, list):
        return []
    out: list[str] = []
    for x in images:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out


def build_shared_image_urls(product: dict, hint_sku: dict) -> list[str]:
    """``{ProductId}_{1..4}`` URLs; same list is copied onto every SKU."""
    prod = product.get("product") or {}
    slug_pid = _slugify(str(prod.get("ProductId", "product")))
    hints = _string_urls(hint_sku.get("images"))
    u0 = hints[0] if hints else ""
    path0 = u0.split("?", 1)[0] if u0 else ""
    if path0 and "/" in path0:
        folder = path0.rsplit("/", 1)[0]
    else:
        folder = (os.getenv("CATALOG_IMAGE_BASE_URL") or _DEFAULT_RAW_BASE).rstrip("/")
    ext = "png"
    if path0:
        m = re.search(r"\.(png|jpg|jpeg|webp)$", path0, re.I)
        if m:
            ext = m.group(1).lower()
            if ext == "jpeg":
                ext = "jpg"
    return [f"{folder}/{slug_pid}_{i}.{ext}" for i in range(1, NUM_IMAGES_PER_PRODUCT + 1)]


def apply_shared_images_to_all_skus(product: dict) -> None:
    skus = product.get("skus")
    if not isinstance(skus, list) or not skus:
        return
    skus_sorted = sorted(skus, key=lambda s: s.get("SkuId", 0))
    shared = build_shared_image_urls(product, skus_sorted[0])
    for sku in skus_sorted:
        sku["images"] = list(shared)


def spec_map(sku: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for sp in sku.get("Specifications") or []:
        if isinstance(sp, dict) and sp.get("Name"):
            out[str(sp["Name"])] = sp.get("Value")
    return out


def pattern_for_product(skus: list[dict]) -> tuple[str, ...] | None:
    names: set[str] = set()
    for sku in skus:
        names.update(spec_map(sku).keys())
    if not names:
        return None
    return tuple(sorted(names))


def most_common(values: list[str]) -> str:
    c = Counter(values)
    return c.most_common(1)[0][0]


def unique_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def pad_sizes(found: list[str], need: int) -> list[str]:
    u = unique_preserve(found)
    for s in DEFAULT_SIZE_POOL:
        if len(u) >= need:
            break
        if s not in u:
            u.append(s)
    while len(u) < need:
        u.append(f"Size {len(u) + 1}")
    return u[:need]


def ensure_distinct_pairs(
    pairs: list[tuple[str, str]], all_a: list[str], all_b: list[str], need: int
) -> list[tuple[str, str]]:
    """Ensure `need` distinct (a,b) pairs."""
    out: list[tuple[str, str]] = []
    used: set[tuple[str, str]] = set()
    for p in pairs:
        if p[0] and p[1] and p not in used:
            used.add(p)
            out.append(p)
        if len(out) >= need:
            return out[:need]
    for a in all_a:
        for b in all_b:
            if (a, b) not in used:
                used.add((a, b))
                out.append((a, b))
                if len(out) >= need:
                    return out[:need]
    i = 0
    while len(out) < need and all_a and all_b:
        a = all_a[i % len(all_a)]
        b = all_b[(i // len(all_a)) % len(all_b)]
        if (a, b) not in used:
            used.add((a, b))
            out.append((a, b))
        i += 1
        if i > need * 50:
            break
    while len(out) < need:
        out.append((f"Option {len(out) + 1}", f"Variant {len(out) + 1}"))
    return out[:need]


def build_sku_name(base: str, pattern: tuple[str, ...], m: dict[str, str]) -> str:
    if pattern == ("Color", "Material", "Size"):
        return f"{base} - {m['Color']} / {m['Size']}"
    if pattern == ("Color", "Dimensions", "Material"):
        return f"{base} - {m['Color']} / {m['Dimensions']}"
    if pattern == ("Color", "Connectivity", "Warranty"):
        return f"{base} - {m['Connectivity']} / {m['Warranty']}"
    if pattern == ("Color", "Material", "Size", "Use Level"):
        return f"{base} - {m['Color']} / {m['Size']} ({m['Use Level']})"
    if pattern == ("Life Stage", "Pet Type", "Weight"):
        return f"{base} - {m['Weight']}"
    if pattern == ("Format", "Strength", "Use"):
        return f"{base} - {m['Strength']}"
    if pattern == ("Age Range", "Material", "Safety"):
        return f"{base} - {m['Age Range']}"
    if pattern == ("Color", "Format", "Pack Size"):
        return f"{base} - {m['Color']} / {m['Pack Size']}"
    if pattern == ("Compatibility", "Material", "Voltage"):
        return f"{base} - {m['Compatibility']} / {m['Voltage']}"
    if pattern == ("Material", "Power Source", "Use"):
        return f"{base} - {m['Power Source']} / {m['Use']}"
    if pattern == ("Form", "Skin/Hair Type", "Volume"):
        return f"{base} - {m['Volume']}"
    if pattern == ("Diet", "Pack Size", "Storage"):
        return f"{base} - {m['Pack Size']}"
    # Fallback: join all spec values in pattern order
    parts = [str(m[k]) for k in pattern if k in m]
    return f"{base} - {' / '.join(parts)}"


def optimize_product(product: dict) -> None:
    skus = product.get("skus")
    if not isinstance(skus, list) or not skus:
        return

    skus_sorted = sorted(skus, key=lambda s: s.get("SkuId", 0))
    pattern = pattern_for_product(skus_sorted)
    if pattern is None:
        return

    rule = PATTERN_RULES.get(pattern)
    if rule is None:
        return

    lock_keys = rule["lock"]
    vary_keys = rule["vary"]

    # Stable spec field order in output (alphabetical, matches pattern tuple)
    order = list(pattern)

    # Canonical locked values (mode across SKUs)
    canonical: dict[str, str] = {}
    for k in lock_keys:
        vals = [spec_map(s)[k] for s in skus_sorted if k in spec_map(s)]
        if vals:
            canonical[k] = most_common(vals)

    n = len(skus_sorted)

    if len(vary_keys) == 1:
        vk = vary_keys[0]
        raw = [spec_map(s).get(vk, "") for s in skus_sorted]
        raw = [str(v) if v is not None else "" for v in raw]
        pool = pad_sizes([v for v in raw if v], n)
        vary_assign = [pool[i] for i in range(n)]
    else:
        v0, v1 = vary_keys[0], vary_keys[1]
        pairs = [
            (str(spec_map(s).get(v0, "") or ""), str(spec_map(s).get(v1, "") or ""))
            for s in skus_sorted
        ]
        all_a = unique_preserve([spec_map(s).get(v0, "") for s in skus_sorted if spec_map(s).get(v0)])
        all_b = unique_preserve([spec_map(s).get(v1, "") for s in skus_sorted if spec_map(s).get(v1)])
        if len(all_a) < 1:
            all_a = ["Option A", "Option B", "Option C"]
        if len(all_b) < 1:
            all_b = ["Option 1", "Option 2", "Option 3"]
        distinct_pairs = ensure_distinct_pairs(pairs, all_a, all_b, n)
        vary_assign = distinct_pairs  # list of tuples

    prod = product.get("product") or {}
    base_name = prod.get("Name") or "Product"

    for i, sku in enumerate(skus_sorted):
        m = dict(canonical)
        if len(vary_keys) == 1:
            m[vary_keys[0]] = vary_assign[i]
        else:
            m[vary_keys[0]], m[vary_keys[1]] = vary_assign[i]

        sku["Specifications"] = [{"Name": k, "Value": m[k]} for k in order if k in m]
        sku["Name"] = build_sku_name(base_name, pattern, m)


def main() -> None:
    if not DATA_FILE.is_file():
        raise SystemExit(f"Missing {DATA_FILE}")

    with DATA_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    products = data.get("products")
    if not isinstance(products, list):
        raise SystemExit("Invalid JSON: expected top-level 'products' list")

    for p in products:
        if isinstance(p, dict):
            optimize_product(p)
            apply_shared_images_to_all_skus(p)

    with DATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Updated {DATA_FILE}")


if __name__ == "__main__":
    main()
