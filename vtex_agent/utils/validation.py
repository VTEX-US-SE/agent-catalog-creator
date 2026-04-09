"""Data validation and normalization utilities."""
import re
from typing import Any, Dict, List, Optional


def normalize_spec_name(name: str) -> str:
    """
    Normalize specification name: capitalize first letter, rest lowercase.
    
    Args:
        name: Specification name to normalize
        
    Returns:
        Normalized specification name
    """
    if not name or not name.strip():
        return name
    name = name.strip()
    # Capitalize first letter, rest lowercase
    if len(name) > 1:
        return name[0].upper() + name[1:].lower()
    else:
        return name.upper()


def normalize_category_name(name: str) -> str:
    """
    Normalize category name: capitalize first letter of each word.
    
    Args:
        name: Category name to normalize
        
    Returns:
        Normalized category name
    """
    if not name or not name.strip():
        return name
    # Title case: first letter of each word capitalized
    return name.strip().title()


def normalize_brand_name(name: str) -> str:
    """
    Normalize brand name: preserve case but trim whitespace.
    
    Args:
        name: Brand name to normalize
        
    Returns:
        Normalized brand name
    """
    if not name:
        return name
    return name.strip()


def extract_product_id(value: Any) -> Optional[int]:
    """
    Extract numeric product ID from various formats.
    
    Args:
        value: Product ID value (can be int, str, etc.)
        
    Returns:
        Product ID as integer, or None if extraction fails
    """
    if value is None:
        return None
    
    if isinstance(value, int):
        return value
    
    if isinstance(value, str):
        # Try to convert directly
        try:
            return int(value.strip())
        except ValueError:
            # Try to extract numbers from string
            numbers = re.findall(r'\d+', value)
            if numbers:
                return int(numbers[0])
    
    return None


def extract_sku_id(value: Any) -> Optional[int]:
    """
    Extract numeric SKU ID from various formats.
    
    Args:
        value: SKU ID value (can be int, str, etc.)
        
    Returns:
        SKU ID as integer, or None if extraction fails
    """
    return extract_product_id(value)  # Same logic


def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Basic JSON schema validation.
    
    Args:
        data: Data to validate
        schema: Schema definition with 'required' and 'optional' keys
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data, dict):
        return False, "Data must be a dictionary"
    
    # Check required fields
    required = schema.get("required", [])
    for field in required:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Check field types (basic validation)
    field_types = schema.get("types", {})
    for field, expected_type in field_types.items():
        if field in data:
            if not isinstance(data[field], expected_type):
                return False, f"Field '{field}' must be of type {expected_type.__name__}"
    
    return True, None


def validate_legacy_site_output(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate Legacy Site agent output JSON.
    
    Args:
        data: Output data to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data, dict):
        return False, "Output must be a dictionary"
    
    required_fields = ["target_url", "products"]
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    if not isinstance(data["products"], list):
        return False, "Products must be a list"
    
    # Validate product structure
    for i, product in enumerate(data["products"]):
        if not isinstance(product, dict):
            return False, f"Product {i} must be a dictionary"
        if "url" not in product:
            return False, f"Product {i} missing 'url' field"
        if "product" not in product:
            return False, f"Product {i} missing 'product' field"
    
    return True, None


def validate_vtex_structure(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate VTEX structure JSON (category tree, specifications, etc.).
    
    Args:
        data: Structure data to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data, dict):
        return False, "Structure must be a dictionary"
    
    # At minimum, should have some structure
    if not data:
        return False, "Structure cannot be empty"
    
    return True, None

