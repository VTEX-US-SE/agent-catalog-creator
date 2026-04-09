"""Sitemap extraction and recursive crawler for finding product URLs."""
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Set, Optional
from xml.etree import ElementTree as ET


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
}


def build_session():
    """Build a requests session with retry logic."""
    s = requests.Session()
    s.headers.update(HEADERS)
    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry = Retry(
            total=5,
            backoff_factor=0.8,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD", "OPTIONS"]
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.mount("http://", HTTPAdapter(max_retries=retry))
    except Exception:
        pass
    return s


def extract_sitemap_urls(base_url: str, session: requests.Session) -> List[str]:
    """
    Try to find and extract URLs from sitemap.xml files.
    
    Returns list of product/page URLs found in sitemaps.
    """
    parsed = urlparse(base_url)
    sitemap_paths = [
        "/sitemap.xml",
        "/sitemap_index.xml",
        "/sitemaps.xml",
        "/robots.txt",  # robots.txt often contains sitemap locations
    ]
    
    all_urls = []
    
    # Try to get sitemap index or main sitemap
    for path in sitemap_paths:
        sitemap_url = f"{parsed.scheme}://{parsed.netloc}{path}"
        try:
            print(f"   🔍 Checking: {sitemap_url}")
            r = session.get(sitemap_url, timeout=15)
            if r.status_code == 200:
                content_type = r.headers.get("content-type", "").lower()
                if "xml" in content_type or path.endswith(".xml"):
                    urls = _parse_sitemap_xml(r.text, session, base_url)
                    all_urls.extend(urls)
                elif "robots.txt" in path:
                    # Extract sitemap URLs from robots.txt
                    sitemap_refs = re.findall(r"Sitemap:\s*(.+)", r.text, re.I)
                    for sitemap_ref in sitemap_refs:
                        urls = _parse_sitemap_xml(session.get(sitemap_ref.strip()).text, session, base_url)
                        all_urls.extend(urls)
                break
        except Exception as e:
            print(f"     ⚠️  Error accessing {sitemap_url}: {e}")
            continue
    
    return list(set(all_urls))  # Deduplicate


def _parse_sitemap_xml(xml_content: str, session: requests.Session, base_url: str) -> List[str]:
    """Parse sitemap XML and extract URLs, handling sitemap indexes."""
    urls = []
    try:
        root = ET.fromstring(xml_content)
        namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        
        # Check if this is a sitemap index
        sitemapindex = root.find("sm:sitemapindex", namespace)
        if sitemapindex is None:
            sitemapindex = root.find("sitemapindex")  # Try without namespace
        
        if sitemapindex is not None:
            # It's a sitemap index, fetch all referenced sitemaps
            for sitemap in sitemapindex.findall("sm:sitemap", namespace) or sitemapindex.findall("sitemap"):
                loc_elem = sitemap.find("sm:loc", namespace) or sitemap.find("loc")
                if loc_elem is not None and loc_elem.text:
                    try:
                        sub_content = session.get(loc_elem.text.strip(), timeout=15).text
                        urls.extend(_parse_sitemap_xml(sub_content, session, base_url))
                    except Exception as e:
                        print(f"     ⚠️  Error fetching sub-sitemap {loc_elem.text}: {e}")
        else:
            # Regular sitemap, extract URLs
            for url_elem in root.findall("sm:url", namespace) or root.findall("url"):
                loc = url_elem.find("sm:loc", namespace) or url_elem.find("loc")
                if loc is not None and loc.text:
                    urls.append(loc.text.strip())
    except ET.ParseError:
        # Fallback: try regex for simple cases
        urls = re.findall(r"<loc>(.+?)</loc>", xml_content)
    
    return urls


def recursive_crawl_pdp_patterns(
    base_url: str,
    session: requests.Session,
    max_pages: int = 100,
    sleep_sec: float = 1.0
) -> List[str]:
    """
    Recursively crawl website to collect ALL URLs.
    
    This function collects all URLs found during crawling without filtering.
    The LLM will later filter and select Product Detail Pages (PDPs).
    """
    parsed = urlparse(base_url)
    domain = f"{parsed.scheme}://{parsed.netloc}"
    
    visited: Set[str] = set()
    all_urls: Set[str] = set()
    queue: List[str] = [base_url]
    
    page_count = 0
    
    while queue and page_count < max_pages:
        current_url = queue.pop(0)
        
        if current_url in visited:
            continue
        
        visited.add(current_url)
        page_count += 1
        
        try:
            print(f"   📄 [{page_count}/{max_pages}] Crawling: {current_url}")
            r = session.get(current_url, timeout=25)
            if r.status_code != 200:
                continue
            
            soup = BeautifulSoup(r.text, "html.parser")
            
            # Add current URL to collection
            all_urls.add(current_url)
            
            # Extract ALL links to crawl (no filtering)
            for a in soup.find_all("a", href=True):
                href = a.get("href")
                if not href:
                    continue
                
                # Normalize URL
                if href.startswith("//"):
                    href = f"{parsed.scheme}:{href}"
                elif href.startswith("/"):
                    href = urljoin(domain, href)
                elif not href.startswith("http"):
                    continue
                
                # Stay within same domain
                if urlparse(href).netloc != parsed.netloc:
                    continue
                
                # Remove fragments and query params for dedup
                clean_href = href.split("#")[0].split("?")[0]
                
                # Add ALL URLs found (no pattern matching)
                all_urls.add(clean_href)
                
                # Add to queue if not visited
                if clean_href not in visited and len(queue) < 500:
                    queue.append(clean_href)
            
            time.sleep(sleep_sec)
            
        except Exception as e:
            print(f"     ⚠️  Error crawling {current_url}: {e}")
            continue
    
    print(f"   ✅ Collected {len(all_urls)} URLs (all URLs, not filtered)")
    return list(all_urls)


def _is_product_page(soup: BeautifulSoup) -> bool:
    """Heuristic to detect if a page is a Product Detail Page."""
    # Check for common product page indicators
    indicators = [
        soup.find("script", type="application/ld+json"),  # JSON-LD Product
        soup.find("meta", {"property": "og:type", "content": re.compile(r"product", re.I)}),
        soup.find("span", {"itemprop": "price"}),
        soup.select_one('[itemtype*="Product"]'),
        soup.select_one('form[action*="cart"], form[action*="add"]'),
        soup.select_one('button[class*="buy"], button[class*="add-to-cart"]'),
    ]
    return any(indicator is not None for indicator in indicators)


def crawl_categories(
    base_categories: List[str],
    session: requests.Session,
    product_url_pattern: Optional[str] = None,
    max_pages_per_category: int = 20,
    sleep_sec: float = 1.2
) -> List[str]:
    """
    Crawl category pages to extract product URLs.
    
    This function crawls through category listing pages (with pagination) to find
    product detail page URLs. It's useful when sitemaps are not available or
    incomplete, and you know the base category URLs.
    
    Args:
        base_categories: List of category base URLs to crawl (e.g., 
            ["https://example.com/category1/", "https://example.com/category2/"])
        session: Requests session with retry logic
        product_url_pattern: Optional regex pattern to match product URLs.
            If None, will use common patterns. Example: r"https?://www\.example\.com/[^/]+-\d+\.html$"
        max_pages_per_category: Maximum number of pages to crawl per category
        sleep_sec: Delay between requests (seconds)
    
    Returns:
        List of unique product URLs found across all categories
    
    Example:
        >>> session = build_session()
        >>> categories = [
        ...     "https://www.dafiti.com.br/calcados-femininos/",
        ...     "https://www.dafiti.com.br/roupas-femininas/"
        ... ]
        >>> urls = crawl_categories(categories, session)
    """
    if not base_categories:
        print("   ⚠️  No base categories provided")
        return []
    
    parsed_base = urlparse(base_categories[0])
    domain = f"{parsed_base.scheme}://{parsed_base.netloc}"
    
    # Build product URL pattern
    if product_url_pattern:
        product_re = re.compile(product_url_pattern)
    else:
        # Generic pattern: URLs with product IDs (e.g., ...-123456.html or /product/123)
        # This is a fallback - ideally the agent should provide a specific pattern
        product_re = re.compile(
            r".*[-\/]\d{5,}\.html?$|"  # URLs ending with product ID
            r".*\/p\/[^\/]+$|"  # /p/slug pattern
            r".*\/product\/[^\/]+$|"  # /product/slug pattern
            r".*\/produto\/[^\/]+$"  # Portuguese /produto/slug
        )
    
    all_product_urls = []
    
    for cat_idx, category_url in enumerate(base_categories, 1):
        print(f"\n   📂 [{cat_idx}/{len(base_categories)}] Crawling category: {category_url}")
        category_products = []
        last_count = -1
        
        for page in range(1, max_pages_per_category + 1):
            # Build paginated URL
            if "?" in category_url:
                page_url = f"{category_url}&page={page}"
            else:
                page_url = f"{category_url}?page={page}"
            
            print(f"      → Page {page}: {page_url}")
            
            try:
                r = session.get(page_url, timeout=25, headers={"Referer": category_url})
                if r.status_code != 200:
                    print(f"         ⚠️  HTTP {r.status_code}, stopping category crawl")
                    break
                
                # Extract product links from page
                page_products = _extract_product_links_from_page(
                    r.text, 
                    category_url, 
                    domain, 
                    product_re
                )
                
                print(f"         Found {len(page_products)} product links on this page")
                
                # If no products found, likely reached end of pagination
                if len(page_products) == 0:
                    print("         No products found, stopping category crawl")
                    break
                
                # Add new products (avoid duplicates)
                for url in page_products:
                    if url not in category_products:
                        category_products.append(url)
                
                # Heuristic: if count doesn't increase for 2 consecutive pages, stop
                if len(category_products) == last_count:
                    print("         No new products found, stopping category crawl")
                    break
                
                last_count = len(category_products)
                time.sleep(sleep_sec)
                
            except Exception as e:
                print(f"         ⚠️  Error crawling page {page}: {e}")
                break
        
        print(f"      ✅ Category complete: {len(category_products)} products found")
        all_product_urls.extend(category_products)
    
    # Deduplicate globally while preserving order
    seen = set()
    unique_urls = []
    for url in all_product_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    print(f"\n   ✅ Total unique products across all categories: {len(unique_urls)}")
    return unique_urls


def _extract_product_links_from_page(
    html: str,
    base_url: str,
    domain: str,
    product_pattern: re.Pattern
) -> List[str]:
    """
    Extract product links from a category listing page HTML.
    
    Args:
        html: HTML content of the page
        base_url: Base URL of the category (for relative URL resolution)
        domain: Domain to filter links (only same-domain links)
        product_pattern: Compiled regex pattern to match product URLs
    
    Returns:
        List of product URLs found on the page
    """
    soup = BeautifulSoup(html, "html.parser")
    parsed_base = urlparse(base_url)
    
    # Collect hrefs from various sources
    hrefs = []
    
    # 1. Standard anchor tags
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        if href:
            hrefs.append(href)
    
    # 2. Images with data-href (common in product cards)
    for img in soup.find_all("img"):
        data_href = img.get("data-href")
        if data_href:
            hrefs.append(data_href)
    
    # 3. Other data attributes that might contain product URLs
    for elem in soup.find_all(attrs={"data-product-url": True}):
        hrefs.append(elem.get("data-product-url"))
    
    # Normalize URLs
    normalized = []
    for href in hrefs:
        if not href:
            continue
        
        # Normalize relative URLs
        if href.startswith("//"):
            href = f"{parsed_base.scheme}:{href}"
        elif href.startswith("/"):
            href = urljoin(domain, href)
        elif not href.startswith("http"):
            continue
        
        # Remove fragments and query params for matching
        clean_href = href.split("#")[0].split("?")[0]
        normalized.append(clean_href)
    
    # Filter by product pattern and domain
    product_urls = []
    for url in normalized:
        parsed = urlparse(url)
        # Only same domain
        if parsed.netloc != parsed_base.netloc:
            continue
        # Match product pattern
        if product_pattern.match(url):
            product_urls.append(url)
    
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for url in product_urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    
    return unique

