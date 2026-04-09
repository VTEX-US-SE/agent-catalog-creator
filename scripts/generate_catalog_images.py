#!/usr/bin/env python3
"""Generate catalog images, upload to GitHub, and patch catalog JSON.

Default behavior creates one image set per product named
``{ProductId}_{view}.png`` (``view`` is 1..NUM_VIEWS_TARGET) and copies the same
URLs into every SKU ``images`` field.

With ``--image-scope sku``, each SKU receives its own image set named
``{ProductId}_{SkuId}_{view}.png``.

Image backends (``--model`` / ``GEMINI_IMAGE_MODEL``):

- **Gemini native image** (e.g. ``gemini-2.5-flash-image`` “Nano Banana”): uses
  ``generate_content`` with ``ImageConfig`` (default aspect ratio 1:1).
- **Imagen** (``imagen-*``): uses ``generate_images`` (separate quota from Gemini image models).
"""

import argparse
import base64
import json
import os
import re
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

import certifi
from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vtex_agent.tools.image_manager import upload_image_to_github

NUM_VIEWS_TARGET = 4


def _slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "item"


def _spec_value(specs: List[Dict[str, Any]], key: str) -> Optional[str]:
    out = None
    key_lower = key.lower()
    for spec in specs:
        name = str(spec.get("Name", "")).strip().lower()
        if name == key_lower:
            out = str(spec.get("Value", "")).strip()
    return out or None


def _build_base_prompt_parts(product: Dict[str, Any], sku: Dict[str, Any]) -> Dict[str, str]:
    product_name = product.get("product", {}).get("Name", "Product")
    categories = product.get("categories", [])
    category = categories[-1]["Name"] if categories else "General Merchandise"
    specs = sku.get("Specifications", [])
    color = _spec_value(specs, "Color") or "neutral tone"
    material = _spec_value(specs, "Material") or "premium material"
    size = _spec_value(specs, "Size") or "standard size"
    sku_name = sku.get("Name", f"{product_name} / {size}")
    key_feature = f"{material} texture and finish"
    brief_description = f"{material} with {color} colorway, SKU variation {sku_name}"
    background = "light gray seamless studio background"

    return {
        "product_name": product_name,
        "category": category,
        "key_feature": key_feature,
        "brief_description": brief_description,
        "background": background,
        "sku_name": sku_name,
    }


def _build_view_prompts(product: Dict[str, Any], sku: Dict[str, Any]) -> List[str]:
    parts = _build_base_prompt_parts(product, sku)
    views = [
        "front view",
        "45-degree profile view",
        "back view",
        f"close-up detail shot of the {parts['key_feature']}",
    ]
    prompts: List[str] = []
    for view in views:
        prompts.append(
            f"A professional high-resolution product photography image for {parts['product_name']} ({parts['sku_name']}). "
            f"The image features a {parts['category']} in a single {view}.\n\n"
            f"The product is a {parts['brief_description']}.\n\n"
            f"Layout: Single isolated product centered on a seamless, minimalist {parts['background']}.\n\n"
            "Lighting & Style: Bright, soft studio lighting with subtle soft shadows to create depth. "
            "Commercial 8k quality, sharp focus, hyper-realistic textures, and a clean aesthetic suitable "
            "for a premium e-commerce website."
        )
    return prompts


def _model_uses_imagen_api(model: str) -> bool:
    return model.lower().startswith("imagen-")


def _blob_to_bytes(blob: object) -> Optional[bytes]:
    if blob is None or blob.data is None:
        return None
    raw = blob.data
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, str):
        return base64.b64decode(raw)
    return None


def _generate_image_bytes(client: genai.Client, model: str, prompt: str) -> bytes:
    if _model_uses_imagen_api(model):
        response = client.models.generate_images(
            model=model,
            prompt=prompt,
            config=types.GenerateImagesConfig(number_of_images=1),
        )
        return response.generated_images[0].image.image_bytes

    # Gemini native image models (e.g. gemini-2.5-flash-image, gemini-3.1-flash-image-preview)
    image_cfg: dict[str, str] = {"aspect_ratio": "1:1"}
    mlow = model.lower()
    if "3.1-flash-image" in mlow or "3-pro-image" in mlow:
        image_cfg["image_size"] = "1K"
    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            image_config=types.ImageConfig(**image_cfg),
        ),
    )
    parts = response.parts or []
    for part in parts:
        if part.inline_data is not None:
            data = _blob_to_bytes(part.inline_data)
            if data:
                return data
    raise RuntimeError(
        f"No image bytes in generate_content response for model={model!r} "
        f"(parts={len(parts)})."
    )


def _url_basename(url: str) -> str:
    return url.rstrip("/").split("/")[-1].split("?")[0]


def _parse_product_view_filename(name: str) -> Optional[tuple[str, int]]:
    """Map basename to (slug_product_id, view_index).

    Supports ``{pid}_{view}.png`` (new) and legacy ``{pid}_{sku}_{view}.png``.
    Slugified ids use hyphens only, so new names split as two segments.
    """
    if not name.lower().endswith(".png"):
        return None
    stem = name[:-4]
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    try:
        view_idx = int(parts[-1])
    except ValueError:
        return None
    slug_pid = parts[0]
    if len(parts) == 2:
        return (slug_pid, view_idx)
    # Legacy: first segment is product slug; last is view; middle is sku slug (ignored).
    return (slug_pid, view_idx)


def _parse_scope_view_filename(
    name: str, image_scope: str
) -> Optional[tuple[str, Optional[str], int]]:
    """Map basename to (slug_product_id, optional_slug_sku_id, view_index)."""
    if not name.lower().endswith(".png"):
        return None
    stem = name[:-4]
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    try:
        view_idx = int(parts[-1])
    except ValueError:
        return None
    slug_pid = parts[0]
    if image_scope == "product":
        return (slug_pid, None, view_idx)
    if len(parts) >= 3:
        return (slug_pid, parts[1], view_idx)
    return None


def _clear_product_level_image_fields(item: Dict[str, Any]) -> None:
    """Remove legacy product-level image lists; URLs live on SKUs only."""
    item.pop("images", None)
    md = item.get("mapped_data")
    if isinstance(md, dict):
        md.pop("images", None)


def _collect_prior_urls(item: Dict[str, Any], skus_to_process: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for u in item.get("images") or []:
        if isinstance(u, str) and u.strip():
            out.append(u.strip())
    for sku in skus_to_process:
        for u in sku.get("images") or []:
            if isinstance(u, str) and u.strip():
                out.append(u.strip())
    return out


def _index_existing_by_product(urls: List[str]) -> Dict[str, Dict[int, str]]:
    """slug_pid -> {view_index: url} (new URLs override legacy for same view)."""
    out: Dict[str, Dict[int, str]] = {}
    for u in urls:
        parsed = _parse_product_view_filename(_url_basename(u))
        if not parsed:
            continue
        pid_slug, vi = parsed
        out.setdefault(pid_slug, {})[vi] = u
    return out


def _index_existing_by_scope(
    urls: List[str], image_scope: str
) -> Dict[tuple[str, Optional[str]], Dict[int, str]]:
    out: Dict[tuple[str, Optional[str]], Dict[int, str]] = {}
    for u in urls:
        parsed = _parse_scope_view_filename(_url_basename(u), image_scope=image_scope)
        if not parsed:
            continue
        slug_pid, slug_sku, view_idx = parsed
        key = (slug_pid, slug_sku)
        out.setdefault(key, {})[view_idx] = u
    return out


def _product_fully_imaged(
    slug_pid: str, existing_by_product: Dict[str, Dict[int, str]]
) -> bool:
    have = existing_by_product.get(slug_pid, {})
    return all(v in have for v in range(1, NUM_VIEWS_TARGET + 1))


def run(args: argparse.Namespace) -> None:
    load_dotenv()
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env")

    with open(args.input, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    products = catalog.get("products", [])
    model = args.model
    client = genai.Client(api_key=api_key)

    started = 0
    uploaded_count = 0
    failed_count = 0
    temp_dir = tempfile.mkdtemp(prefix="catalog_img_")

    for p_idx, item in enumerate(products):
        if p_idx < args.start_product:
            continue
        if args.max_products and started >= args.max_products:
            break

        skus = item.get("skus", [])
        if not skus:
            continue

        skus_to_process = skus
        if args.max_skus_per_product:
            skus_to_process = skus[: args.max_skus_per_product]

        prior_urls = _collect_prior_urls(item, skus_to_process)
        existing_by_scope = _index_existing_by_scope(
            prior_urls, image_scope=args.image_scope
        )

        product_id = str(item.get("product", {}).get("ProductId", f"p{p_idx + 1}"))
        slug_pid = _slugify(product_id)

        if args.image_scope == "product":
            if args.skip_existing and _product_fully_imaged(
                slug_pid, {k[0]: v for k, v in existing_by_scope.items() if k[1] is None}
            ):
                have = dict(existing_by_scope.get((slug_pid, None), {}))
                urls = [have[v] for v in range(1, NUM_VIEWS_TARGET + 1)]
                for sku in skus_to_process:
                    sku["images"] = list(urls)
                _clear_product_level_image_fields(item)
                continue

            skus_sorted = sorted(skus_to_process, key=lambda s: s.get("SkuId", 0))
            canonical_sku = skus_sorted[0]
            have = dict(existing_by_scope.get((slug_pid, None), {}))
            urls: List[str] = []

            try:
                prompts = _build_view_prompts(item, canonical_sku)

                for view_idx, prompt in enumerate(prompts, start=1):
                    if args.skip_existing and view_idx in have:
                        urls.append(have[view_idx])
                        continue

                    image_bytes = _generate_image_bytes(client, model, prompt)
                    filename = f"{slug_pid}_{view_idx}.png"
                    local_path = os.path.join(temp_dir, filename)
                    with open(local_path, "wb") as img_file:
                        img_file.write(image_bytes)

                    raw_url = upload_image_to_github(
                        image_path=local_path,
                        filename=filename,
                        repo_path=args.github_repo_path,
                        github_branch=os.getenv("GITHUB_BRANCH", "main"),
                    )
                    if raw_url:
                        urls.append(raw_url)
                        have[view_idx] = raw_url
                        existing_by_scope.setdefault((slug_pid, None), {})[view_idx] = raw_url
                        uploaded_count += 1
                    else:
                        failed_count += 1

                    if args.sleep_s > 0:
                        time.sleep(args.sleep_s)

            except Exception as exc:
                failed_count += 1
                print(f"[WARN] Product #{p_idx + 1} ({product_id!r}) failed: {exc}")
                have = dict(existing_by_scope.get((slug_pid, None), {}))
                urls = [have[i] for i in range(1, NUM_VIEWS_TARGET + 1) if i in have]
                if len(urls) < NUM_VIEWS_TARGET:
                    print(
                        f"[WARN] Product #{p_idx + 1} incomplete after error "
                        f"({len(urls)} URLs)."
                    )

            if len(urls) >= NUM_VIEWS_TARGET:
                urls = urls[:NUM_VIEWS_TARGET]
            else:
                have = dict(existing_by_scope.get((slug_pid, None), {}))
                urls = [have[i] for i in range(1, NUM_VIEWS_TARGET + 1) if i in have]

            for sku in skus_to_process:
                sku["images"] = list(urls)
        else:
            for sku in sorted(skus_to_process, key=lambda s: s.get("SkuId", 0)):
                sku_id = str(sku.get("SkuId", "sku"))
                slug_sku = _slugify(sku_id)
                key = (slug_pid, slug_sku)
                have = dict(existing_by_scope.get(key, {}))
                urls: List[str] = []

                try:
                    prompts = _build_view_prompts(item, sku)
                    for view_idx, prompt in enumerate(prompts, start=1):
                        if args.skip_existing and view_idx in have:
                            urls.append(have[view_idx])
                            continue

                        image_bytes = _generate_image_bytes(client, model, prompt)
                        filename = f"{slug_pid}_{slug_sku}_{view_idx}.png"
                        local_path = os.path.join(temp_dir, filename)
                        with open(local_path, "wb") as img_file:
                            img_file.write(image_bytes)

                        raw_url = upload_image_to_github(
                            image_path=local_path,
                            filename=filename,
                            repo_path=args.github_repo_path,
                            github_branch=os.getenv("GITHUB_BRANCH", "main"),
                        )
                        if raw_url:
                            urls.append(raw_url)
                            have[view_idx] = raw_url
                            existing_by_scope.setdefault(key, {})[view_idx] = raw_url
                            uploaded_count += 1
                        else:
                            failed_count += 1

                        if args.sleep_s > 0:
                            time.sleep(args.sleep_s)
                except Exception as exc:
                    failed_count += 1
                    print(
                        f"[WARN] Product #{p_idx + 1} ({product_id!r}) SKU {sku_id!r} failed: {exc}"
                    )
                    have = dict(existing_by_scope.get(key, {}))
                    urls = [have[i] for i in range(1, NUM_VIEWS_TARGET + 1) if i in have]

                if len(urls) >= NUM_VIEWS_TARGET:
                    urls = urls[:NUM_VIEWS_TARGET]
                else:
                    have = dict(existing_by_scope.get(key, {}))
                    urls = [have[i] for i in range(1, NUM_VIEWS_TARGET + 1) if i in have]
                sku["images"] = list(urls)
        _clear_product_level_image_fields(item)

        started += 1
        if started % args.save_every == 0:
            with open(args.output, "w", encoding="utf-8") as out:
                json.dump(catalog, out, ensure_ascii=False, indent=2)
            print(
                f"[SAVE] processed_products={started} uploaded_images={uploaded_count} failed_images={failed_count}"
            )

    with open(args.output, "w", encoding="utf-8") as out:
        json.dump(catalog, out, ensure_ascii=False, indent=2)
    print(
        f"[DONE] processed_products={started} uploaded_images={uploaded_count} failed_images={failed_count}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and upload catalog images.")
    parser.add_argument("--input", default="json_generated_catalog.json")
    parser.add_argument("--output", default="json_generated_catalog.json")
    # Default: Gemini 2.5 Flash Image (“Nano Banana”) — separate quota from Imagen.
    # Imagen: https://ai.google.dev/gemini-api/docs/imagen
    # Native image: https://ai.google.dev/gemini-api/docs/image-generation
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image"),
        help=(
            "Image model: Gemini native (e.g. gemini-2.5-flash-image) or Imagen (e.g. imagen-4.0-generate-001)."
        ),
    )
    parser.add_argument("--github-repo-path", default="images")
    parser.add_argument("--start-product", type=int, default=0)
    parser.add_argument("--max-products", type=int, default=0)
    parser.add_argument("--max-skus-per-product", type=int, default=0)
    parser.add_argument(
        "--image-scope",
        choices=["product", "sku"],
        default="product",
        help="Generate one gallery per product (default) or per SKU.",
    )
    parser.add_argument("--sleep-s", type=float, default=0.4)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
