"""Google Gemini integration for mapping extracted HTML to VTEX Catalog Schema."""
import os
import json
import re
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Try to use the new google-genai SDK, fallback to legacy google-generativeai
try:
    from google import genai as genai_sdk
    from google.genai import types
    USE_NEW_SDK = True
except ImportError:
    try:
        import google.generativeai as genai_sdk
        USE_NEW_SDK = False
    except ImportError:
        raise ImportError(
            "Neither 'google-genai' nor 'google-generativeai' is installed. "
            "Please install one: pip install google-genai"
        )

load_dotenv()


def _retry_with_exponential_backoff(
    func,
    max_retries: int = 5,
    initial_delay: float = 2.0,  # Increased from 1.0 to 2.0 seconds
    max_delay: float = 120.0,  # Increased from 60.0 to 120.0 seconds
    backoff_factor: float = 2.0,
    pre_request_delay: float = 0.5  # Small delay before first request
):
    """
    Retry a function with exponential backoff for 429 (rate limit) errors.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds (default: 2.0)
        max_delay: Maximum delay in seconds (default: 120.0)
        backoff_factor: Multiplier for exponential backoff
        pre_request_delay: Small delay before first request to avoid immediate rate limits
        
    Returns:
        Result of the function call
        
    Raises:
        Last exception if all retries fail
    """
    # Small delay before first request to avoid hitting rate limits immediately
    if pre_request_delay > 0:
        time.sleep(pre_request_delay)
    
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            
            # Check if it's a rate limit error (429)
            error_str = str(e).lower()
            error_type = type(e).__name__
            
            # Check for ClientError from google.genai SDK
            is_client_error = "ClientError" in error_type or "clienterror" in error_type.lower()
            
            # Check error message and attributes
            is_rate_limit = (
                "429" in error_str or
                "rate limit" in error_str or
                "quota" in error_str or
                "too many requests" in error_str or
                "resource exhausted" in error_str or
                "resource_exhausted" in error_str
            )
            
            # Check for status code in various places
            if hasattr(e, 'status_code') and e.status_code == 429:
                is_rate_limit = True
            
            # Check for error code in ClientError structure
            if hasattr(e, 'error') and isinstance(e.error, dict):
                error_code = e.error.get('code') or e.error.get('status')
                if error_code == 429 or str(error_code) == "429":
                    is_rate_limit = True
            
            # Check if ClientError contains 429 in its representation
            if is_client_error and ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)):
                is_rate_limit = True
            
            if not is_rate_limit:
                # Not a rate limit error, re-raise immediately
                raise
            
            if attempt < max_retries:
                # Calculate delay with exponential backoff
                wait_time = min(delay, max_delay)
                print(f"     ⚠️  Rate limit error (429). Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries + 1})...")
                time.sleep(wait_time)
                delay *= backoff_factor
            else:
                # Max retries reached
                print(f"     ❌ Rate limit error persisted after {max_retries + 1} attempts")
                print(f"     💡 Consider waiting a few minutes before retrying, or check your API quota")
                raise
    
    # Should never reach here, but just in case
    if last_exception:
        raise last_exception


# VTEX Catalog Schema structure
VTEX_SCHEMA = {
    "Department": {
        "required": ["Name"],
        "optional": ["Active", "MenuHome", "AdWordsRemarketingCode", "Description"]
    },
    "Category": {
        "required": ["Name", "FatherCategoryId"],
        "optional": ["Title", "Description", "Keywords", "Active"]
    },
    "Brand": {
        "required": ["Name"],
        "optional": ["Active", "Text", "Keywords", "SiteTitle"]
    },
    "Product": {
        "required": ["Name", "CategoryId", "BrandId"],
        "optional": [
            "Description", "ShortDescription", "ReleaseDate", "KeyWords",
            "Title", "IsActive", "ShowWithoutStock", "Score"
        ]
    },
    "SKU": {
        "required": ["ProductId", "Name", "EAN", "IsActive"],
        "optional": [
            "RefId", "Height", "Width", "Length", "Weight",
            "Price", "ListPrice", "CostPrice"
        ]
    },
    "Specification": {
        "required": ["Name", "CategoryId", "FieldId"],
        "optional": ["Value"]
    }
}


def preprocess_html(html_content: str) -> str:
    """
    Preprocess HTML to reduce size while preserving product information.
    Removes scripts, styles, and unnecessary elements.
    
    Args:
        html_content: Raw HTML content
        
    Returns:
        Cleaned HTML content
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style tags
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        
        # Remove comments
        from bs4 import Comment
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment.extract()
        
        # Remove empty elements
        for tag in soup.find_all():
            if not tag.get_text(strip=True) and not tag.find_all(['img', 'input', 'meta', 'link']):
                tag.decompose()
        
        # Keep only relevant sections for product extraction
        # Preserve: head (for meta tags), body content, product-related sections
        return str(soup)
    except Exception as e:
        # If preprocessing fails, return original
        print(f"   ⚠️  HTML preprocessing failed: {e}, using original HTML")
        return html_content


def initialize_gemini(api_key: Optional[str] = None):
    """
    Initialize Gemini API client.
    Uses global endpoint by default to avoid 429 rate limit errors.
    See: https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429
    """
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Set it in .env file or pass as parameter."
        )
    
    # Get model name from environment variable (default: gemini-2.0-flash)
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    
    # Get base URL from environment variable
    # Default to global endpoint for Vertex AI to avoid 429 errors
    # Global endpoint is recommended per Google Cloud documentation
    base_url = os.getenv("GEMINI_BASE_URL")
    
    # If no base URL is set, use global endpoint for Vertex AI
    # For Vertex AI, using global endpoint helps avoid 429 errors
    if not base_url:
        # Use global endpoint - this is the recommended approach per Google docs
        # The SDK will use the appropriate global endpoint automatically
        base_url = None
    
    if USE_NEW_SDK:
        # Use new google-genai SDK
        # If base_url is None, SDK uses default global endpoint
        if base_url:
            client = genai_sdk.Client(
                api_key=api_key,
                http_options=types.HttpOptions(base_url=base_url)
            )
        else:
            # Use default global endpoint (recommended to avoid 429 errors)
            client = genai_sdk.Client(api_key=api_key)
        
        # Return client and model name tuple for new SDK
        return (client, model_name)
    else:
        # Fallback to legacy google-generativeai SDK
        # Legacy SDK uses global endpoint by default
        genai_sdk.configure(api_key=api_key)
        return genai_sdk.GenerativeModel(model_name)


def extract_to_vtex_schema(
    html_content: str,
    url: str,
    api_key: Optional[str] = None,
    custom_instructions: Optional[str] = None
) -> Dict[str, Any]:
    """
    Use Gemini to extract product data from HTML and map to VTEX schema.
    
    Args:
        html_content: Raw HTML content from product page
        url: Source URL of the product page
        api_key: Optional Gemini API key (uses env var if not provided)
        custom_instructions: Optional custom instructions to append to the prompt
    
    Returns structured data matching VTEX Catalog API requirements.
    """
    model = initialize_gemini(api_key)
    
    # Preprocess HTML to reduce size ONLY if file is greater than 200k characters
    # Otherwise, use complete HTML to preserve all details
    original_size = len(html_content)
    
    if original_size > 200000:  # Only preprocess if greater than 200k characters
        cleaned_html = preprocess_html(html_content)
        cleaned_size = len(cleaned_html)
        reduction = ((original_size - cleaned_size) / original_size * 100) if original_size > 0 else 0
        html_to_send = cleaned_html
        print(f"     📉 HTML size reduced: {original_size:,} → {cleaned_size:,} chars ({reduction:.1f}% reduction)")
    else:
        html_to_send = html_content
        print(f"     📄 Using complete HTML ({original_size:,} chars)")
    
    # Build the prompt with cleaned HTML
    prompt = f"""You are a VTEX Catalog Integration Specialist. Extract product information from the following HTML and map it to the VTEX Catalog Schema.

HTML Content:
{html_to_send}

Source URL: {url}

VTEX Schema Requirements:
{json.dumps(VTEX_SCHEMA, indent=2)}

EXTRACTION INSTRUCTIONS - Follow these guidelines carefully:

1. PRODUCT ID (ProductId):
   - Look for data attributes: data-product-id, data-productid, product-id, productId, product_id
   - Check hidden inputs: input[name*="product"][type="hidden"], input[id*="product"]
   - Extract from URL patterns: numbers in URL path, query parameters
   - Check script tags with JSON-LD or product data
   - Look for IDs in class names or IDs like "product-12345"
   - If ProductId is found, use it for SkuId and RefId as well (unless different SKU ID is found)

2. PRODUCT NAME:
   - Extract from h1 tags, preferably with itemprop="name" or class containing "product-name", "product-title"
   - Preserve capitalization and special characters
   - Use the most prominent product title on the page

3. DESCRIPTION:
   - Look for meta description tags
   - Find description in: div.description, span.description, section.product-details, .product-description
   - Check for itemprop="description"
   - Extract full text, not truncated versions
   - If multiple descriptions found, use the longest/most complete one

4. KEYWORDS:
   - Extract from meta keywords tag
   - Generate from: product name, brand, categories, specifications
   - Include: product type, brand, main categories, key specifications
   - Format as comma-separated string

5. TITLE (SEO):
   - Prefer meta title tag if available
   - Otherwise: "Product Name | Category | Brand | Site Name"
   - Include category hierarchy for better SEO

6. SKU EXTRACTION:
   - SkuId: Use ProductId if no separate SKU ID found (data-sku-id, sku-id, etc.)
   - EAN: Extract from barcode, EAN fields, or use ProductId if EAN not available
   - RefId: Use ProductId or SKU code found in HTML
   - Name: Extract SKU-specific name (variation details like size, color, dimensions)
     If no variation name found, use a descriptive name based on specifications
   - IsActive: Always set to FALSE (SKUs should be inactive by default)
   - activeIfPossible: Always set to TRUE (indicates SKU can be activated if needed)
   - Price: Extract from price elements, data-price attributes, or price input fields
     Look for: .price, .product-price, [data-price], input[name*="price"], span.price
     Convert to float number (remove currency symbols, commas)
   - ListPrice: Same as Price if no separate list price found

7. SPECIFICATIONS:
   - Look for specification tables: <table>, <dl> (definition lists), <ul> with labels
   - Find key-value pairs in: div.specifications, div.attributes, section.product-details
   - Extract ALL specifications found, including:
     * Material, Dimensões, Bitola, Comprimento, Acabamento
     * Peso, Embalagem, Origem, Norma, Classe Resistência
     * Any other technical specifications or attributes
   - Format: {{"Name": "Specification Label", "Value": "Specification Value"}}
   - Do NOT leave specifications array empty if any are found on the page

8. IMAGES:
   - CRITICAL: Extract ONLY product images, NOT banners, logos, icons, or decorative images
   - Look for product image galleries, carousels, main product photos, zoom images
   - Prioritize HIGH-RESOLUTION images (look for URLs with: 1200Wx1200H, 800x800, large, high-res)
   - Extract from: product image galleries, carousels, main product photo containers, zoom/lightbox images
   - Sources: img tags in product galleries, data-src attributes in image carousels, product image containers
   - EXCLUDE: site logos, banners, icons, social media images, decorative backgrounds, navigation images
   - Convert relative URLs to absolute URLs using the base URL
   - Include all relevant product images (different angles, details, variations)
   - Prefer full-size images over thumbnail versions
   - If no clear product images are found, return empty array (do not include non-product images)

9. CATEGORIES:
   - CRITICAL: Extract categories from HTML CONTENT, not just URL path
   - Primary sources (in order of priority):
     1. Breadcrumb navigation: nav.breadcrumb, ol.breadcrumb, .breadcrumbs, .breadcrumb-nav
     2. Category navigation menus visible on the page
     3. Category links in product detail sections
     4. Meta tags: meta[property="product:category"], meta[name="category"]
     5. Structured data: JSON-LD with category information
     6. URL path structure (as fallback if HTML doesn't contain category info)
   - Map each level to appropriate category Level (1 = top level, 2 = subcategory, etc.)
   - If categories are not in URL but are in breadcrumbs/navigation, use those instead
   - Extract the full category hierarchy from the page structure

10. BRAND:
    - Extract from: brand name in product title, brand logo alt text, data-brand attribute
    - Check meta tags: meta[property="product:brand"], meta[name="brand"]

IMPORTANT: 
- Do NOT return null values unless the field is truly not found after thorough search
- Product IsActive MUST always be TRUE (products should always be active)
- SKU IsActive MUST always be FALSE (SKUs should always be inactive)
- SKU activeIfPossible MUST always be TRUE (indicates SKU can be activated)
- Extract ALL specifications - do not skip any technical details
- Use ProductId for SkuId and RefId if no separate SKU identifiers are found
- Ensure prices are numbers (floats), not strings
- Prioritize high-resolution images over thumbnails

Please extract and return a JSON object with the following structure:
{{
    "categories": [
        // Array of categories extracted from HTML content (breadcrumbs, navigation, etc.)
        // Priority: 1) Breadcrumbs, 2) Navigation menus, 3) Meta tags, 4) URL path (fallback)
        // Example: ["Elementos de Fixação", "Parafusos", "Parafusos Francês"]
        // First item is the top-level category (Level 1)
        // Subsequent items are subcategories (Level 2, 3, etc.)
        {{"Name": "category name from HTML content", "Level": 1}},
        {{"Name": "subcategory name from HTML content", "Level": 2}},
        // Add more levels as needed based on category hierarchy found in HTML
    ],
    "brand": {{
        "Name": "brand name"
    }},
    "product": {{
        "Name": "product title/name (preserve capitalization)",
        "ProductId": "product ID - MUST be extracted, do not return null",
        "Description": "full product description - extract from description elements",
        "ShortDescription": "brief description (first 200 chars of Description)",
        "KeyWords": "comma-separated keywords from meta tags or generated from product data",
        "Title": "SEO title - prefer meta title, otherwise format as: Name | Category | Brand",
        "IsActive": true,
        "ShowWithoutStock": true
    }},
    "skus": [
        {{
            "Name": "SKU name (variation details or descriptive name based on specs)",
            "SkuId": "SKU ID - use ProductId if no separate SKU ID found",
            "EAN": "EAN code - use ProductId if EAN not available",
            "IsActive": false,
            "activeIfPossible": true,
            "RefId": "reference ID - use ProductId or SKU code",
            "Price": 0.0,
            "ListPrice": 0.0
        }}
    ],
    "images": [
        // Extract ONLY product images (galleries, carousels, main product photos)
        // EXCLUDE: banners, logos, icons, decorative images, navigation images
        // Prioritize high-resolution versions (1200Wx1200H or larger)
        // Convert to absolute URLs using the base URL
        "full URL to high-resolution product image"
    ],
    "specifications": [
        // Extract ALL specifications from the page
        // Look in tables, definition lists, specification sections
        // Include ALL technical details found
        {{
            "Name": "specification label (e.g., 'Acabamento', 'Bitola', 'Material')",
            "Value": "specification value"
        }}
    ]
}}"""
    
    # Append custom instructions if provided
    if custom_instructions:
        prompt += f"""

CRITICAL: CUSTOM EXTRACTION RULES - PRIORITIZE THESE INSTRUCTIONS
====================================================================
{custom_instructions}
====================================================================

CRITICAL: The above custom extraction rules specify EXACT CSS selectors and HTML element paths 
to use for extracting product information. These rules MUST take PRIORITY over all default extraction logic.
"""

    # Append custom instructions if provided
    if custom_instructions:
        prompt += f"""

CRITICAL: CUSTOM EXTRACTION RULES - PRIORITIZE THESE INSTRUCTIONS
====================================================================
{custom_instructions}
====================================================================

CRITICAL: The above custom extraction rules specify EXACT CSS selectors and HTML element paths 
to use for extracting product information. These rules MUST take PRIORITY over all default extraction logic.
"""

    prompt += """

FINAL REMINDERS:
- ProductId MUST be extracted - search thoroughly in data attributes, inputs, URLs, and script tags
- If ProductId is found (e.g., "10010801"), use it for SkuId, EAN, and RefId unless different values are found
- Description should be the full technical/product description, not just the short description
- KeyWords should be extracted from meta tags or generated from product name, brand, and categories
- Extract ALL specifications from tables, lists, or specification sections - do not leave empty
- Price should be extracted as a number (float) - search in price elements, data attributes, or input fields
- Product IsActive MUST always be TRUE (products should always be active)
- SKU IsActive MUST always be FALSE (SKUs should always be inactive)
- SKU activeIfPossible MUST always be TRUE (indicates SKU can be activated)
- SKU Name should be descriptive (e.g., dimensions, color, or variation details)
- Images should be high-resolution URLs (prefer 1200Wx1200H or larger sizes)
- Do NOT return null for ProductId, SkuId, EAN, RefId, Price, or ListPrice if any product identifier is found
- If price is not found, set Price and ListPrice to 0.0 (not null)

CRITICAL JSON FORMATTING REQUIREMENTS:
- ALL string values MUST be properly escaped (use \\" for quotes inside strings, \\n for newlines)
- Do NOT include unescaped quotes, newlines, or special characters in string values
- Ensure ALL strings are properly closed with closing quotes
- Do NOT include HTML content or raw text in JSON string values - extract only the text content
- If a value contains quotes or special characters, properly escape them or truncate the value
- Return ONLY valid JSON - no explanatory text before or after the JSON object

Return the JSON response now:"""

    def _call_gemini_api():
        """Inner function to call Gemini API (for retry logic)."""
        if USE_NEW_SDK and isinstance(model, tuple):
            # New SDK: (client, model_name)
            client, model_name = model
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            # New SDK response structure may differ - check for text attribute or candidates
            if hasattr(response, 'text'):
                return response.text.strip()
            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                # Fallback: try to get text from candidates
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    return ''.join(part.text for part in candidate.content.parts if hasattr(part, 'text')).strip()
                else:
                    return str(response).strip()
            else:
                return str(response).strip()
        else:
            # Legacy SDK: GenerativeModel instance
            response = model.generate_content(prompt)
            return response.text.strip()
    
    try:
        # Call API with exponential backoff retry for 429 errors
        text = _retry_with_exponential_backoff(_call_gemini_api)
        
        # Extract JSON from response (may be wrapped in markdown)
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        # Try to find JSON object boundaries if response contains extra text
        # Look for first { and last } to extract just the JSON
        first_brace = text.find('{')
        if first_brace != -1:
            # Find the matching closing brace
            brace_count = 0
            last_brace = -1
            for i in range(first_brace, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_brace = i
                        break
            if last_brace != -1:
                text = text[first_brace:last_brace + 1]
        
        # Clean up common JSON issues
        # Remove any trailing commas before closing braces/brackets
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        result = json.loads(text)
        return result
        
    except json.JSONDecodeError as e:
        print(f"   ⚠️  Error parsing Gemini response: {e}")
        try:
            # Try to extract and fix common JSON issues
            print(f"   Attempting to fix JSON parsing errors...")
            
            error_pos = getattr(e, 'pos', None)
            error_msg = str(e)
            
            # Handle unterminated strings
            if "Unterminated string" in error_msg and error_pos:
                print(f"   Detected unterminated string at position {error_pos}, attempting to fix...")
                
                # Find the start of the problematic string
                # Look backwards from error_pos to find the opening quote
                start_pos = error_pos
                quote_char = None
                for i in range(error_pos - 1, max(0, error_pos - 1000), -1):
                    if text[i] in ['"', "'"]:
                        # Check if it's escaped
                        if i > 0 and text[i-1] == '\\':
                            continue
                        quote_char = text[i]
                        start_pos = i
                        break
                
                if quote_char and start_pos < error_pos:
                    # Find where the string should end (next unescaped quote or end of value)
                    # Look for the next quote, comma, colon, or closing brace/bracket
                    end_pos = error_pos
                    for i in range(error_pos, min(len(text), error_pos + 5000)):
                        if text[i] == quote_char:
                            # Check if it's escaped
                            if i > 0 and text[i-1] == '\\':
                                continue
                            end_pos = i + 1
                            break
                        elif text[i] in [',', ':', '}', ']', '\n']:
                            # String might be cut off, try to close it here
                            # Insert closing quote before this character
                            text = text[:i] + quote_char + text[i:]
                            end_pos = i + 1
                            break
                    
                    if end_pos == error_pos:
                        # Couldn't find end, try to close at a safe position
                        # Look for next comma, colon, or closing brace
                        for i in range(error_pos, min(len(text), error_pos + 1000)):
                            if text[i] in [',', ':', '}', ']']:
                                text = text[:i] + quote_char + text[i:]
                                break
                        else:
                            # No safe position found, just close the string
                            text = text[:error_pos] + quote_char + text[error_pos:]
            
            # Try to find JSON object more aggressively
            first_brace = text.find('{')
            if first_brace != -1:
                # Try to extract up to the error position and fix it
                if error_pos and error_pos < len(text):
                    # Try to find a valid JSON structure before the error
                    potential_json = text[first_brace:error_pos]
                    # Try to close any open structures
                    open_braces = potential_json.count('{') - potential_json.count('}')
                    open_brackets = potential_json.count('[') - potential_json.count(']')
                    
                    # Count open strings
                    open_strings = 0
                    in_string = False
                    escape_next = False
                    for char in potential_json:
                        if escape_next:
                            escape_next = False
                            continue
                        if char == '\\':
                            escape_next = True
                            continue
                        if char in ['"', "'"]:
                            if not in_string:
                                in_string = True
                                open_strings += 1
                            else:
                                in_string = False
                                open_strings -= 1
                    
                    # Try to complete the JSON
                    fixed_json = potential_json
                    if in_string:
                        fixed_json += '"'  # Close open string
                    fixed_json += '}' * open_braces
                    fixed_json += ']' * open_brackets
                    
                    try:
                        result = json.loads(fixed_json)
                        print(f"   ✅ Successfully fixed JSON parsing error")
                        return result
                    except json.JSONDecodeError as e2:
                        # Try one more time with the full text after fixes
                        try:
                            # Remove any remaining problematic characters
                            # Escape unescaped newlines in strings
                            fixed_text = re.sub(r'(?<!\\)"(?=.*")', r'\\"', text[:error_pos] + '"' + text[error_pos:])
                            result = json.loads(fixed_text)
                            print(f"   ✅ Successfully fixed JSON parsing error (method 2)")
                            return result
                        except:
                            pass
            
            # Show error context for debugging
            if error_pos:
                start = max(0, error_pos - 200)
                end = min(len(text), error_pos + 200)
                context = text[start:end]
                print(f"   Error context (char {error_pos}):")
                print(f"   ...{context}...")
                # Try to save problematic response for debugging
                try:
                    from ..utils.state_manager import STATE_DIR
                    debug_file = STATE_DIR / "debug_response.json"
                    debug_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                    print(f"   💾 Full response saved to: {debug_file}")
                except:
                    pass
            else:
                print(f"   Response preview: {text[:500]}...")
        except Exception as fix_error:
            print(f"   Could not fix JSON error: {fix_error}")
        raise
    except Exception as e:
        # Show concise error message without full traceback
        error_msg = str(e)
        if "429" in error_msg or "resource exhausted" in error_msg.lower():
            print(f"   ⚠️  Rate limit error (429). Please wait and retry.")
            print(f"   💡 Tip: Use global endpoint and implement exponential backoff retry.")
        else:
            # Show only the error type and brief message, not full traceback
            error_type = type(e).__name__
            brief_msg = error_msg[:200] if len(error_msg) > 200 else error_msg
            print(f"   ⚠️  Error calling Gemini API: {error_type}: {brief_msg}")
        raise


def analyze_structure_from_sample(
    sample_products: list,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze multiple product samples to understand the overall structure.
    
    Returns aggregated structure with categories, brands, and common specifications.
    """
    model = initialize_gemini(api_key)
    
    prompt = f"""Analyze the following product samples and provide a summary of the catalog structure:

{json.dumps(sample_products, indent=2, ensure_ascii=False)[:30000]}

Return a JSON object with:
{{
    "departments": ["list of unique departments"],
    "categories": [
        {{"Name": "category name", "Department": "parent department"}}
    ],
    "brands": ["list of unique brand names"],
    "specification_groups": ["list of common specification group names"],
    "total_products": {len(sample_products)},
    "product_patterns": {{
        "has_variations": true/false,
        "variation_types": ["Color", "Size", ...],
        "common_fields": ["list of fields present in all products"]
    }}
}}

Return JSON only:"""

    def _call_gemini_api():
        """Inner function to call Gemini API (for retry logic)."""
        if USE_NEW_SDK and isinstance(model, tuple):
            # New SDK: (client, model_name)
            client, model_name = model
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            # New SDK response structure may differ - check for text attribute or candidates
            if hasattr(response, 'text'):
                return response.text.strip()
            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                # Fallback: try to get text from candidates
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    return ''.join(part.text for part in candidate.content.parts if hasattr(part, 'text')).strip()
                else:
                    return str(response).strip()
            else:
                return str(response).strip()
        else:
            # Legacy SDK: GenerativeModel instance
            response = model.generate_content(prompt)
            return response.text.strip()
    
    try:
        # Call API with exponential backoff retry for 429 errors
        text = _retry_with_exponential_backoff(_call_gemini_api)
        
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        return json.loads(text)
    except Exception as e:
        # Show concise error message
        error_msg = str(e)
        if "429" in error_msg or "resource exhausted" in error_msg.lower():
            print(f"   ⚠️  Rate limit error (429) in structure analysis.")
        else:
            error_type = type(e).__name__
            brief_msg = error_msg[:200] if len(error_msg) > 200 else error_msg
            print(f"   ⚠️  Error in structure analysis: {error_type}: {brief_msg}")
        return {}

