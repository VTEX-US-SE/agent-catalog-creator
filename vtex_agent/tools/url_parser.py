"""URL parsing utilities for extracting category structure."""
import re
from urllib.parse import urlparse
from typing import List, Dict, Any


def parse_category_tree_from_url(url: str) -> List[Dict[str, Any]]:
    """
    Parse URL structure to extract category hierarchy.
    
    Example:
        URL: /p/elementos-de-fixacao/parafusos/parafuso-sextavado...
        Returns: [
            {"Name": "Elementos de Fixação", "Level": 1},
            {"Name": "Parafusos", "Level": 2}
        ]
    
    Args:
        url: Product URL
        
    Returns:
        List of categories ordered from top-level to leaf
    """
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    
    # Common patterns: /p/category1/category2/product or /category1/category2/product
    # Remove product slug (usually last segment with product name)
    segments = [s for s in path.split("/") if s and s not in ["p", "product", "produto", "item"]]
    
    categories = []
    for i, segment in enumerate(segments, 1):
        # Skip if it looks like a product ID or code (ends with numbers/letters like 10010801)
        if len(segment) > 20 or re.match(r".*-\d{5,}$", segment):
            # Likely product slug, stop here
            break
        
        # Convert URL segment to readable category name
        # Replace hyphens with spaces and title case
        category_name = segment.replace("-", " ").replace("_", " ").title()
        categories.append({
            "Name": category_name,
            "Level": i
        })
    
    return categories

