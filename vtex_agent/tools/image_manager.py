"""Image extraction from HTML pages and image URL utilities."""
import re
import os
import base64
import requests
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from ..utils.logger import get_agent_logger

load_dotenv()

# Initialize logger for image processing
logger = get_agent_logger("image_manager")


def extract_high_res_images(html_content: str, base_url: str) -> List[str]:
    """
    Extract high-resolution product images from HTML.
    
    Returns list of full URLs to high-res images.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content, "html.parser")
    image_urls = []
    
    def _normalize_url(url: str) -> str:
        """Normalize image URL to full path."""
        if not url:
            return ""
        if url.startswith("//"):
            return f"https:{url}"
        if url.startswith("/"):
            return urljoin(base_url, url)
        return url.split("?")[0].split("#")[0]  # Remove query params
    
    # 1) JSON-LD structured data
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            import json
            data = json.loads(script.string or "")
            if isinstance(data, dict) and data.get("@type") == "Product":
                images = data.get("image", [])
                if isinstance(images, str):
                    images = [images]
                for img in images:
                    if isinstance(img, dict):
                        img = img.get("url") or img.get("@id") or ""
                    if img:
                        image_urls.append(_normalize_url(img))
        except Exception:
            pass
    
    # 2) og:image meta tags
    for meta in soup.find_all("meta", {"property": "og:image"}):
        url = meta.get("content")
        if url:
            image_urls.append(_normalize_url(url))
    
    # 3) High-res image selectors (product galleries) - expanded list
    selectors = [
        "img[data-testid='image-gallery-preview']",
        "img[data-testid='image-gallery-thumbnail']",
        "img.product-image",
        "img[itemprop='image']",
        ".product-gallery img",
        ".product-images img",
        ".product-image-gallery img",
        ".product-photos img",
        ".product-carousel img",
        ".product-slider img",
        "[class*='product-image'] img",
        "[class*='product-gallery'] img",
        "[class*='product-photo'] img",
        "[id*='product-image'] img",
        "[id*='product-gallery'] img",
        "img[data-image]",
        "img[data-product-image]",
        "img[data-src]",
        "img[data-lazy]",
        "img[data-lazy-src]",
    ]
    
    for selector in selectors:
        try:
            for img in soup.select(selector):
                # Check srcset for high-res versions
                srcset = img.get("srcset", "")
                if srcset:
                    # Extract highest resolution from srcset
                    parts = srcset.split(",")
                    for part in parts:
                        url = part.strip().split()[0]
                        if url and ("x" in part or "w" in part.lower()):
                            image_urls.append(_normalize_url(url))
                
                # Check data attributes for image URLs
                for attr in ["data-image", "data-product-image", "data-src", "data-lazy", "data-lazy-src"]:
                    data_url = img.get(attr)
                    if data_url:
                        image_urls.append(_normalize_url(data_url))
                
                # Regular src
                src = img.get("src")
                if src:
                    # Replace thumbnail dimensions with full size if pattern exists
                    # e.g., .../thumb_200x200/... -> .../full/...
                    src = re.sub(r"(thumb|thumbnail|small|medium)[_\-\/]?\d+x\d+", "full", src, flags=re.I)
                    image_urls.append(_normalize_url(src))
        except Exception:
            # Skip selector if it causes issues
            continue
    
    # 3b) Generic image search - look for all images in product-related containers
    product_containers = soup.select(
        "[class*='product'], [id*='product'], [class*='item'], [id*='item'], "
        "[data-product], [data-item], .gallery, .carousel, .slider"
    )
    for container in product_containers:
        for img in container.find_all("img"):
            src = img.get("src") or img.get("data-src") or img.get("data-lazy") or img.get("data-image")
            if src:
                image_urls.append(_normalize_url(src))
    
    # 4) Picture source elements (responsive images)
    for source in soup.find_all("source", srcset=True):
        srcset = source.get("srcset", "")
        parts = srcset.split(",")
        for part in parts:
            url = part.strip().split()[0]
            if url:
                image_urls.append(_normalize_url(url))
    
    # Deduplicate and filter
    seen = set()
    final_urls = []
    for url in image_urls:
        if not url:
            continue
        # Normalize URL for comparison (remove query params, fragments)
        normalized = url.split("?")[0].split("#")[0].lower()
        if normalized in seen:
            continue
        
        # Check if it looks like an image URL
        # Accept URLs with image extensions OR URLs that contain image-related paths
        is_image = (
            any(ext in url.lower() for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg"]) or
            any(keyword in url.lower() for keyword in ["/image", "/img", "/photo", "/picture", "/media", "/product"])
        )
        
        # Exclude common non-product images
        exclude_keywords = ["logo", "icon", "favicon", "banner", "ad", "advertisement", "placeholder"]
        if any(keyword in url.lower() for keyword in exclude_keywords):
            continue
        
        if is_image:
            seen.add(normalized)
            final_urls.append(url)
    
    return final_urls[:10]  # Limit to 10 images


def download_image(image_url: str, output_path: str) -> bool:
    """
    Download an image from URL to local file.
    
    Args:
        image_url: URL of the image to download
        output_path: Local path where to save the image
        
    Returns:
        True if successful, False otherwise
    """
    print(f"       📥 Step 1: Downloading image from legacy site...")
    logger.debug(f"Downloading image from URL: {image_url}")
    logger.debug(f"Output path: {output_path}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        logger.debug(f"Sending HTTP request to {image_url}")
        response = requests.get(image_url, headers=headers, timeout=30)
        logger.debug(f"Response status: {response.status_code}")
        response.raise_for_status()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        file_size = len(response.content)
        logger.debug(f"Saving {file_size:,} bytes to {output_path}")
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        logger.info(f"Download successful: {file_size:,} bytes from {image_url}")
        print(f"       ✅ Download successful")
        return True
    except requests.exceptions.Timeout as e:
        logger.error(f"Download failed: Request timeout (30s) for {image_url}: {str(e)}")
        print(f"       ❌ Download failed")
        return False
    except requests.exceptions.HTTPError as e:
        status_code = response.status_code if 'response' in locals() else 'unknown'
        logger.error(f"Download failed: HTTP {status_code} for {image_url}: {str(e)}")
        print(f"       ❌ Download failed")
        return False
    except Exception as e:
        logger.error(f"Download failed: {type(e).__name__} for {image_url}: {str(e)}")
        print(f"       ❌ Download failed")
        return False


def upload_image_to_github(
    image_path: str,
    filename: str,
    repo_path: str = "images",
    github_token: Optional[str] = None,
    github_repo: Optional[str] = None,
    github_branch: str = "main"
) -> Optional[str]:
    """
    Upload an image to GitHub repository and return the raw GitHub URL.
    
    Args:
        image_path: Local path to the image file
        filename: Name for the file in GitHub (e.g., "10010801_1.jpg")
        repo_path: Path within the repository (e.g., "images" or "products/images")
        github_token: GitHub personal access token (or from env GITHUB_TOKEN)
        github_repo: Repository in format "username/repo" (or from env GITHUB_REPO)
        github_branch: Branch name (default: "main")
        
    Returns:
        Raw GitHub URL if successful, None otherwise
    """
    github_token = github_token or os.getenv("GITHUB_TOKEN")
    github_repo = github_repo or os.getenv("GITHUB_REPO")
    
    if not github_token or not github_repo:
        raise ValueError(
            "GitHub credentials required. Set GITHUB_TOKEN and GITHUB_REPO in .env"
        )
    
    # Parse GitHub repo URL if full URL is provided
    # Accept formats: "owner/repo" or "https://github.com/owner/repo" or "https://github.com/owner/repo.git"
    if github_repo.startswith("http"):
        # Extract owner/repo from URL
        parsed = urlparse(github_repo)
        path_parts = [p for p in parsed.path.split("/") if p]
        if len(path_parts) >= 2:
            github_repo = f"{path_parts[0]}/{path_parts[1].replace('.git', '')}"
        elif len(path_parts) == 1:
            raise ValueError(
                f"Invalid GitHub repo format. Expected 'owner/repo' or full URL, got: {github_repo}"
            )
    
    print(f"       📤 Step 2: Uploading to GitHub...")
    logger.debug(f"Uploading to GitHub - Repo: {github_repo}, Path: {repo_path}/{filename}")
    
    # Read image file
    try:
        logger.debug(f"Reading image file: {image_path}")
        with open(image_path, "rb") as f:
            image_content = f.read()
        file_size = len(image_content)
        logger.debug(f"File read: {file_size:,} bytes")
    except Exception as e:
        logger.error(f"Error reading image file {image_path}: {e}")
        print(f"       ❌ Error reading image file")
        return None
    
    # Encode to base64
    logger.debug(f"Encoding to base64...")
    image_base64 = base64.b64encode(image_content).decode("utf-8")
    logger.debug(f"Encoded: {len(image_base64):,} characters")
    
    # GitHub API endpoint
    api_url = f"https://api.github.com/repos/{github_repo}/contents/{repo_path}/{filename}"
    logger.debug(f"API URL: {api_url}")
    
    headers = {
        "Authorization": f"token {github_token[:10]}..." if len(github_token) > 10 else f"token ***",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Check if file already exists
    logger.debug(f"Checking if file exists in repository...")
    check_response = requests.get(api_url, headers={
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    })
    sha = None
    if check_response.status_code == 200:
        existing_file = check_response.json()
        sha = existing_file.get("sha")
        logger.debug(f"File exists, will update (SHA: {sha[:10]}...)")
    elif check_response.status_code == 404:
        logger.debug(f"File does not exist, will create new file")
    else:
        logger.warning(f"Unexpected status checking file: {check_response.status_code}")
    
    # Prepare data
    data = {
        "message": f"Upload image: {filename}",
        "content": image_base64,
        "branch": github_branch
    }
    
    if sha:
        data["sha"] = sha  # Update existing file
    
    # Upload to GitHub
    try:
        logger.debug(f"Uploading to GitHub API...")
        response = requests.put(api_url, json=data, headers={
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }, timeout=30)
        logger.debug(f"Response status: {response.status_code}")
        response.raise_for_status()
        
        # Generate raw GitHub URL
        raw_url = f"https://raw.githubusercontent.com/{github_repo}/{github_branch}/{repo_path}/{filename}"
        logger.info(f"Upload successful: {raw_url}")
        print(f"       ✅ Upload successful")
        return raw_url
    except requests.exceptions.HTTPError as e:
        status_code = response.status_code if 'response' in locals() else 'unknown'
        logger.error(f"Upload failed: HTTP {status_code} for {filename}")
        if hasattr(e, "response") and e.response is not None:
            try:
                error_data = e.response.json()
                error_msg = error_data.get('message', 'Unknown error')
                logger.error(f"GitHub API error: {error_msg}")
                if 'documentation_url' in error_data:
                    logger.debug(f"Docs: {error_data['documentation_url']}")
            except:
                logger.error(f"GitHub API response: {e.response.text[:200]}")
        print(f"       ❌ Upload failed")
        return None
    except Exception as e:
        logger.error(f"Upload failed: {type(e).__name__} for {filename}: {str(e)}")
        if hasattr(e, "response") and e.response is not None:
            try:
                error_data = e.response.json()
                logger.error(f"GitHub API error: {error_data.get('message', 'Unknown error')}")
            except:
                logger.error(f"GitHub API response: {e.response.text[:200]}")
        print(f"       ❌ Upload failed")
        return None


def process_and_upload_images_to_github(
    image_urls: List[str],
    sku_id: int,
    repo_path: str = "images",
    github_token: Optional[str] = None,
    github_repo: Optional[str] = None,
    github_branch: str = "main",
    temp_dir: str = "/tmp/vtex_images"
) -> List[Dict[str, Any]]:
    """
    Prepare image URLs for VTEX association without uploading to GitHub.
    
    Args:
        image_urls: List of image URLs to process
        sku_id: SKU ID for naming (format: {sku_id}_{sequence}.jpg)
        repo_path: Path within GitHub repository
        github_token: GitHub token (or from env)
        github_repo: GitHub repo (or from env)
        github_branch: Branch name
        temp_dir: Temporary directory for downloaded images
        
    Returns:
        List of dicts with image info: [{"url": image_url, "name": filename, "sequence": n}, ...]
    """
    uploaded_images = []
    
    for sequence, image_url in enumerate(image_urls, start=1):
        # Determine file extension from URL or default to jpg
        parsed_url = urlparse(image_url)
        # Get the path without query parameters
        path_without_query = parsed_url.path.split('?')[0]
        
        # Extract extension, but clean it up
        ext = os.path.splitext(path_without_query)[1] or ""
        
        # Remove dimension suffixes like -1200Wx1200H, -800x600, etc.
        # Pattern: -[number][WwHh]x[number][WwHh] or -[number]x[number]
        ext = re.sub(r'-\d+[WwHh]?x\d+[WwHh]?', '', ext, flags=re.I)
        
        # Normalize to standard image extensions
        ext_lower = ext.lower()
        if ext_lower in ['.jpg', '.jpeg']:
            ext = '.jpg'
        elif ext_lower == '.png':
            ext = '.png'
        elif ext_lower == '.webp':
            ext = '.webp'
        elif ext_lower == '.gif':
            ext = '.gif'
        elif ext_lower == '.svg':
            ext = '.svg'
        else:
            # Default to .jpg if extension is not recognized or empty
            ext = '.jpg'
        
        # Try to keep the source file name when available.
        source_name = os.path.basename(path_without_query)
        if source_name and "." in source_name:
            filename = source_name
        else:
            # Fallback filename expected by the rest of the pipeline.
            filename = f"{sku_id}_{sequence}{ext}"

        print(f"\n     [{sequence}/{len(image_urls)}] Processing image: {filename}")
        print(f"     {'='*60}")

        if not image_url:
            uploaded_images.append({
                "url": None,
                "name": filename,
                "sequence": sequence,
                "original_url": image_url,
                "status": "failed",
                "error": "Empty image URL"
            })
            logger.warning(f"Skipping image {filename}: empty URL")
            print(f"       ❌ Empty URL")
            continue

        # No upload step: use existing URL directly.
        uploaded_images.append({
            "url": image_url,
            "name": filename,
            "sequence": sequence,
            "original_url": image_url,
            "status": "uploaded"
        })
        logger.info(f"Image {filename} prepared using existing URL (no GitHub upload)")
        print(f"       ✅ Using existing image URL")
    
    return uploaded_images

