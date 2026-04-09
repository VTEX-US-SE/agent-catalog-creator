"""VTEX Catalog API client for creating categories, brands, products, and SKUs."""
import requests
from typing import Dict, Any, List, Optional
import os
from dotenv import load_dotenv

load_dotenv()


class VTEXClient:
    """Client for VTEX Catalog API operations."""
    
    def __init__(
        self,
        account_name: Optional[str] = None,
        environment: str = "vtexcommercestable",
        app_key: Optional[str] = None,
        app_token: Optional[str] = None
    ):
        self.account_name = account_name or os.getenv("VTEX_ACCOUNT_NAME")
        self.environment = environment
        self.app_key = app_key or os.getenv("VTEX_APP_KEY")
        self.app_token = app_token or os.getenv("VTEX_APP_TOKEN")
        
        if not all([self.account_name, self.app_key, self.app_token]):
            raise ValueError(
                "VTEX credentials required. Set VTEX_ACCOUNT_NAME, "
                "VTEX_APP_KEY, and VTEX_APP_TOKEN in .env"
            )
        
        self.base_url = (
            f"https://{self.account_name}.{self.environment}.com.br"
        )
        self.headers = {
            "X-VTEX-API-AppKey": self.app_key,
            "X-VTEX-API-AppToken": self.app_token,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> requests.Response:
        """Make API request with error handling."""
        url = f"{self.base_url}/api/catalog/{endpoint}"
        
        try:
            response = requests.request(
                method=method,
                url=url,
                json=data,
                params=params,
                headers=self.headers,
                timeout=30
            )
            
            # Log errors with more detail for debugging
            if not response.ok:
                error_msg = response.text[:300] if response.text else "No error message"
                print(f"   ⚠️  VTEX API Error [{response.status_code}] {method} {endpoint}")
                print(f"       Response: {error_msg}")
                # For 404s, show the full URL for debugging
                if response.status_code == 404:
                    print(f"       Full URL: {url}")
                    if params:
                        print(f"       Params: {params}")
                    if data:
                        print(f"       Data keys: {list(data.keys())}")
            
            return response
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request exception: {e}")
            # Return a mock response object to avoid breaking callers
            class MockResponse:
                status_code = 500
                text = str(e)
                def json(self):
                    return {}
                def raise_for_status(self):
                    pass
            return MockResponse()
    
    # ========== CATEGORY OPERATIONS ==========
    
    def create_department(self, name: str, active: bool = True) -> Dict[str, Any]:
        """Create a department (root category). Always created with IsActive, ShowInStoreFront, ActiveStoreFrontLink true."""
        endpoint = "pvt/category"
        data = {
            "Name": name,
            "AdWordsRemarketingCode": None,
            "Description": None,
            "Active": True,
            "MenuHome": True,
            "IsActive": True,
            "ShowInStoreFront": True,
            "ActiveStoreFrontLink": True,
            "GlobalCategoryId": 1,
        }
        response = self._request("POST", endpoint, data=data)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 400 and "already exists" in response.text.lower():
            # Department exists, get it and ensure active/storefront flags are set
            existing = self.get_category_by_name(name)
            if existing:
                cat_id = existing.get("Id")
                if cat_id is not None:
                    self.update_category(cat_id, is_active=True, show_in_store_front=True, active_store_front_link=True, global_category_id=1)
                return existing
        response.raise_for_status()
        return {}
    
    def get_category_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get category by name (searches all categories)."""
        categories = self.list_categories()
        if not categories:
            return None
        for cat in categories:
            if isinstance(cat, dict) and cat.get("Name") == name:
                return cat
        return None
    
    def create_category(
        self,
        name: str,
        father_category_id: Optional[int] = None,
        title: Optional[str] = None,
        active: bool = True
    ) -> Dict[str, Any]:
        """Create a category. Always created with IsActive, ShowInStoreFront, ActiveStoreFrontLink true."""
        endpoint = "pvt/category"
        data = {
            "Name": name,
            "FatherCategoryId": father_category_id,
            "Title": title or name,
            "Description": None,
            "Keywords": None,
            "Active": True,
            "IsActive": True,
            "ShowInStoreFront": True,
            "ActiveStoreFrontLink": True,
            "GlobalCategoryId": 1,
        }
        response = self._request("POST", endpoint, data=data)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 400 and "already exists" in response.text.lower():
            existing = self.get_category_by_name(name)
            if existing:
                cat_id = existing.get("Id")
                if cat_id is not None:
                    self.update_category(cat_id, is_active=True, show_in_store_front=True, active_store_front_link=True, global_category_id=1)
                return existing
            return {}
        response.raise_for_status()
    
    def update_category(
        self,
        category_id: int,
        is_active: Optional[bool] = None,
        show_in_store_front: Optional[bool] = None,
        active_store_front_link: Optional[bool] = None,
        global_category_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update category flags. Fetches current category, merges flags, PUTs back."""
        endpoint = f"pvt/category/{category_id}"
        response = self._request("GET", endpoint)
        if response.status_code != 200:
            response.raise_for_status()
            return {}
        data = response.json()
        if is_active is not None:
            data["IsActive"] = is_active
            data["Active"] = is_active
        if show_in_store_front is not None:
            data["ShowInStoreFront"] = show_in_store_front
        if active_store_front_link is not None:
            data["ActiveStoreFrontLink"] = active_store_front_link
        if global_category_id is not None:
            data["GlobalCategoryId"] = global_category_id
        put_response = self._request("PUT", endpoint, data=data)
        if put_response.status_code == 200:
            return put_response.json()
        put_response.raise_for_status()
        return {}
    
    def list_categories(self) -> List[Dict[str, Any]]:
        """List all categories. Returns a list of category dicts (handles list or wrapped response)."""
        endpoint = "pvt/category"
        response = self._request("GET", endpoint)
        if response.status_code != 200:
            return []
        data = response.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Some APIs return { "data": [...], "range": {...} or similar
            for key in ("data", "items", "categories", "CategoryTree", "value"):
                if isinstance(data.get(key), list):
                    return data[key]
        return []
    
    # ========== BRAND OPERATIONS ==========
    
    def create_brand(
        self,
        name: str,
        active: bool = True,
        site_title: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a brand."""
        endpoint = "pvt/brand"
        data = {
            "Name": name,
            "Active": active,
            "Text": None,
            "Keywords": None,
            "SiteTitle": site_title or name
        }
        response = self._request("POST", endpoint, data=data)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 400 and "already exists" in response.text.lower():
            # Try to get existing brand
            brands = self.list_brands()
            for brand in brands:
                if isinstance(brand, dict) and brand.get("Name") == name:
                    return brand
        response.raise_for_status()
        return {}
    
    def list_brands(self) -> List[Dict[str, Any]]:
        """List all brands."""
        endpoint = "pvt/brand"
        response = self._request("GET", endpoint)
        if response.status_code == 200:
            return response.json()
        return []
    
    # ========== SPECIFICATION OPERATIONS ==========
    
    def create_specification_group(
        self,
        group_name: str,
        category_id: int
    ) -> Dict[str, Any]:
        """Create a specification group (required before creating fields)."""
        # Check if group already exists
        existing_groups = self.list_specification_groups(category_id)
        for group in existing_groups:
            if isinstance(group, dict) and group.get("Name") == group_name:
                print(f"         ℹ️  Specification group '{group_name}' already exists (ID: {group.get('Id')})")
                return group
        
        # Try different endpoint variations
        endpoints_to_try = [
            f"pvt/specification/group",
            f"pvt/specification/group/{category_id}",
            f"pvt/category/{category_id}/specification/group"
        ]
        
        data = {
            "Name": group_name,
            "CategoryId": category_id
        }
        
        for endpoint in endpoints_to_try:
            response = self._request("POST", endpoint, data=data)
            if response.status_code == 200:
                return response.json()
            elif response.status_code in [400, 409]:
                # Might already exist, try to get it
                existing_groups = self.list_specification_groups(category_id)
                for group in existing_groups:
                    if isinstance(group, dict) and group.get("Name") == group_name:
                        return group
        
        # If all endpoints fail, return empty (groups might be optional in some VTEX versions)
        print(f"         ⚠️  Could not create specification group '{group_name}', continuing without it")
        return {}
    
    def list_specification_groups(self, category_id: int) -> List[Dict[str, Any]]:
        """List all specification groups for a category."""
        endpoints_to_try = [
            ("GET", f"pvt/specification/group", {"CategoryId": category_id}),
            ("GET", f"pvt/specification/group/{category_id}", None),
            ("GET", f"pvt/category/{category_id}/specification/group", None)
        ]
        
        for method, endpoint, params in endpoints_to_try:
            response = self._request(method, endpoint, params=params)
            if response.status_code == 200:
                result = response.json()
                return result if isinstance(result, list) else []
        
        return []
    
    def create_specification_field(
        self,
        field_name: str,
        category_id: int,
        field_type: str = "Text",
        is_required: bool = False,
        group_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a specification field.
        
        Based on VTEX API documentation, the correct endpoint is:
        POST /api/catalog/pvt/specification/field
        """
        # First, check if field already exists to avoid duplicates
        # If listing fails (404), we'll skip this check and try to create anyway
        try:
            existing_fields = self.list_specification_fields(category_id)
            for field in existing_fields:
                if isinstance(field, dict) and field.get("Name") == field_name:
                    print(f"         ℹ️  Specification field '{field_name}' already exists (ID: {field.get('Id')})")
                    return field
        except Exception as e:
            # If listing fails, continue with creation attempt
            self.logger.debug(f"Could not list existing fields (will try to create anyway): {e}")
        
        # Build data payload according to VTEX API spec
        data = {
            "Name": field_name,
            "CategoryId": category_id,
            "FieldTypeId": self._get_field_type_id(field_type),
            "IsRequired": is_required,
            "IsStockKeepingUnit": False,  # Usually False unless it's a variation field
            "IsFilter": True,
            "IsOnProductDetails": True
        }
        
        # Add group ID if provided (optional)
        if group_id:
            data["GroupId"] = group_id
        
        # Try multiple endpoint and method combinations
        # VTEX APIs sometimes use PUT for create operations
        attempts = [
            ("PUT", "pvt/specification/field", data),  # PUT is common in VTEX
            ("POST", "pvt/specification/field", data),
            ("PUT", f"pvt/specification/field/{category_id}", data),
            ("POST", f"pvt/specification/field/{category_id}", data),
            ("PUT", f"pvt/category/{category_id}/specification/field", data),
            ("POST", f"pvt/category/{category_id}/specification/field", data),
        ]
        
        last_error = None
        for method, endpoint, payload in attempts:
            response = self._request(method, endpoint, data=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"         ✅ Created specification field '{field_name}' (ID: {result.get('Id')}) using {method} {endpoint}")
                return result
            
            # Store error for reporting
            if response.status_code not in [400, 409]:  # Don't overwrite validation errors
                last_error = (response.status_code, response.text[:200], method, endpoint)
            
            # Handle validation/duplicate errors
            if response.status_code in [400, 409]:
                error_text = response.text.lower()
                if "already exists" in error_text or "duplicate" in error_text:
                    # Field might exist, try to find it
                    existing_fields = self.list_specification_fields(category_id)
                    for field in existing_fields:
                        if isinstance(field, dict) and field.get("Name") == field_name:
                            print(f"         ℹ️  Field found after creation attempt (ID: {field.get('Id')})")
                            return field
                else:
                    # Validation error - log details
                    print(f"         ⚠️  Validation error creating field '{field_name}': {response.text[:300]}")
                    return {}
        
        # Log error details
        if last_error:
            status, error_text, method, endpoint = last_error
            print(f"         ⚠️  Failed to create specification field '{field_name}'. Status: {status}")
            print(f"         ⚠️  Last attempt: {method} {endpoint}")
            print(f"         ⚠️  Response: {error_text}")
        else:
            print(f"         ⚠️  Failed to create specification field '{field_name}' - all methods returned validation errors")
        
        # Don't raise error, return empty dict to allow continuation
        return {}
    
    def _get_field_type_id(self, field_type: str) -> int:
        """Map field type string to VTEX field type ID."""
        mapping = {
            "Text": 1,
            "Number": 2,
            "Toggle": 3,
            "Combo": 4,
            "Radio": 5,
            "Color": 6,
            "Date": 7
        }
        return mapping.get(field_type, 1)
    
    def list_specification_fields(self, category_id: int) -> List[Dict[str, Any]]:
        """
        List all specification fields for a category.
        
        Tries multiple endpoint variations since VTEX API structure can vary.
        """
        attempts = [
            ("GET", "pvt/specification/field", {"CategoryId": category_id}),
            ("GET", f"pvt/specification/field/{category_id}", None),
            ("GET", f"pvt/category/{category_id}/specification/field", None),
            ("GET", "pvt/specification/field", None),  # Try without category filter
        ]
        
        for method, endpoint, params in attempts:
            try:
                response = self._request(method, endpoint, params=params)
                if response.status_code == 200:
                    result = response.json()
                    return result if isinstance(result, list) else []
            except Exception:
                continue
        
        # If all attempts fail, return empty list (will try to create anyway)
        return []
    
    def set_product_specification(
        self,
        product_id: int,
        field_id: int,
        field_value: str,
        field_type: str = "Text"
    ) -> Dict[str, Any]:
        """
        Set a specification value for a product.
        
        Based on VTEX API documentation:
        PUT /api/catalog/pvt/products/{productId}/specification
        
        For Text fields: use "Text" parameter, FieldValueId = null
        For Combo/Radio/Checkbox: use "FieldValueId" parameter, Text = null
        """
        endpoint = f"pvt/products/{product_id}/specification"
        
        # Build data based on field type
        data = {
            "FieldId": field_id
        }
        
        # For Text fields, use Text parameter
        # For Combo/Radio/Checkbox, we'd need FieldValueId (not implemented yet)
        if field_type in ["Text", "Number", "Date"]:
            data["FieldValueId"] = None
            data["Text"] = str(field_value)
        else:
            # For Combo/Radio/Checkbox, we'd need to create/get FieldValueId first
            # For now, try Text as fallback
            data["FieldValueId"] = None
            data["Text"] = str(field_value)
        
        response = self._request("PUT", endpoint, data=data)
        
        if response.status_code == 200:
            return response.json()
        
        # Log error but don't raise - allow continuation
        if response.status_code != 200:
            print(f"         ⚠️  Failed to set specification. Status: {response.status_code}")
            print(f"         ⚠️  Response: {response.text[:200]}")
        
        return {}
    
    # ========== PRODUCT OPERATIONS ==========
    
    def create_product(
        self,
        name: str,
        category_id: int,
        brand_id: int,
        description: Optional[str] = None,
        short_description: Optional[str] = None,
        is_active: bool = True,
        is_visible: bool = True,
        show_without_stock: bool = True,
        product_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a product.
        
        Args:
            name: Product name
            category_id: Category ID
            brand_id: Brand ID
            description: Product description
            short_description: Short description
            is_active: Whether product is active
            is_visible: Whether product is visible (always True when creating)
            show_without_stock: Show product even without stock
            product_id: Optional product ID to use (if not provided, VTEX will assign one)
        """
        endpoint = "pvt/product"
        data = {
            "Name": name,
            "CategoryId": category_id,
            "BrandId": brand_id,
            "Description": description or "",
            "ShortDescription": short_description or (description[:200] if description else ""),
            "ReleaseDate": None,
            "KeyWords": None,
            "Title": name,
            "IsActive": is_active,
            "IsVisible": is_visible,
            "ShowWithoutStock": show_without_stock,
            "Score": None
        }
        
        # If product_id is provided, include it in the data
        if product_id is not None:
            data["Id"] = product_id
        
        response = self._request("POST", endpoint, data=data)
        if response.status_code == 200:
            return response.json()
        
        # Handle 409 Conflict - product already exists
        if response.status_code == 409 and product_id is not None:
            # Try to get the existing product
            try:
                existing_product = self.get_product(product_id)
                if existing_product:
                    print(f"   ℹ️  Product already exists, using existing product (ID: {product_id})")
                    # Update IsActive flag to ensure Display on website is enabled
                    need_update = (
                        (is_active and not existing_product.get("IsActive", False)) or
                        (is_visible and not existing_product.get("IsVisible", False))
                    )
                    if need_update:
                        try:
                            self.update_product(product_id, is_active=True, is_visible=True)
                            if not existing_product.get("IsActive", False):
                                print(f"   ✓ Updated product IsActive flag to True")
                            if not existing_product.get("IsVisible", False):
                                print(f"   ✓ Updated product IsVisible flag to True")
                        except Exception as update_error:
                            print(f"   ⚠️  Could not update product flags: {update_error}")
                    return existing_product
            except Exception as e:
                # If we can't get the product, log but don't raise - return empty dict to allow continuation
                print(f"   ⚠️  Product {product_id} already exists but could not retrieve it: {e}")
                print(f"   ℹ️  Continuing with existing product ID: {product_id}")
                # Return a minimal product dict with the ID so the workflow can continue
                return {"Id": product_id, "Name": "Existing Product"}
        
        # Only raise for non-409 errors
        if response.status_code != 409:
            response.raise_for_status()
        else:
            # 409 without product_id - return empty dict to allow continuation
            print(f"   ⚠️  Product creation returned 409 Conflict but no product_id provided")
            return {}
    
    def get_product(self, product_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a product by ID.
        
        Args:
            product_id: Product ID
            
        Returns:
            Product data or None if not found
        """
        endpoint = f"pvt/product/{product_id}"
        response = self._request("GET", endpoint)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None
        response.raise_for_status()
        return None
    
    def update_product(
        self,
        product_id: int,
        is_active: Optional[bool] = None,
        is_visible: Optional[bool] = None,
        show_without_stock: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Update a product's IsActive, IsVisible and ShowWithoutStock flags.
        
        Args:
            product_id: Product ID
            is_active: Whether product is active (Display on website)
            is_visible: Whether product is visible
            show_without_stock: Show product even without stock
            
        Returns:
            Updated product data
        """
        # First get the current product data
        current_product = self.get_product(product_id)
        if not current_product:
            raise ValueError(f"Product {product_id} not found")
        
        # Update only the fields that are provided
        if is_active is not None:
            current_product["IsActive"] = is_active
        if is_visible is not None:
            current_product["IsVisible"] = is_visible
        if show_without_stock is not None:
            current_product["ShowWithoutStock"] = show_without_stock
        
        # Update the product
        endpoint = f"pvt/product/{product_id}"
        response = self._request("PUT", endpoint, data=current_product)
        
        if response.status_code == 200:
            return response.json()
        
        response.raise_for_status()
        return {}
    
    # ========== SKU OPERATIONS ==========
    
    def get_sku(self, sku_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a SKU by ID.
        
        Args:
            sku_id: SKU ID
            
        Returns:
            SKU data or None if not found
        """
        endpoint = f"pvt/stockkeepingunit/{sku_id}"
        response = self._request("GET", endpoint)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None
        # Don't raise for other errors, just return None
        return None
    
    def create_sku(
        self,
        product_id: int,
        name: str,
        ean: str,
        is_active: bool = False,
        ref_id: Optional[str] = None,
        price: Optional[float] = None,
        list_price: Optional[float] = None,
        package_height: Optional[float] = None,
        package_width: Optional[float] = None,
        package_length: Optional[float] = None,
        package_weight: Optional[float] = None,
        height: Optional[float] = None,
        width: Optional[float] = None,
        length: Optional[float] = None,
        weight: Optional[float] = None,
        sku_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a SKU.
        
        Args:
            product_id: Product ID this SKU belongs to
            name: SKU name
            ean: EAN code
            is_active: Whether SKU is active (default False; VTEX requires files/components before activating)
            ref_id: Reference ID
            price: SKU price
            list_price: List price
            package_height: Package height dimension
            package_width: Package width dimension
            package_length: Package length dimension
            package_weight: Package weight
            height: Product height dimension (unpackaged)
            width: Product width dimension (unpackaged)
            length: Product length dimension (unpackaged)
            weight: Product weight (unpackaged)
            sku_id: Optional SKU ID to use (if not provided, VTEX will assign one)
        """
        endpoint = f"pvt/stockkeepingunit"
        data = {
            "ProductId": product_id,
            "Name": name,
            "EAN": ean,
            "IsActive": is_active,
            "RefId": ref_id
        }
        
        # Add package dimensions if provided
        if package_height is not None:
            data["PackagedHeight"] = package_height
        if package_width is not None:
            data["PackagedWidth"] = package_width
        if package_length is not None:
            data["PackagedLength"] = package_length
        if package_weight is not None:
            data["PackagedWeightKg"] = package_weight
        
        # Add unpackaged dimensions if provided
        if height is not None:
            data["Height"] = height
        if width is not None:
            data["Width"] = width
        if length is not None:
            data["Length"] = length
        if weight is not None:
            data["WeightKg"] = weight
        
        # If sku_id is provided, include it in the data
        if sku_id is not None:
            data["Id"] = sku_id
        
        response = self._request("POST", endpoint, data=data)
        if response.status_code == 200:
            sku_data = response.json()
            # Note: Price is NOT set here - it should be set after images are added
            # This allows the correct order: Create SKU > Add images > Add price > Add inventory
            return sku_data
        
        # Handle 409 Conflict - SKU already exists
        if response.status_code == 409 and sku_id is not None:
            # Try to get the existing SKU
            try:
                existing_sku = self.get_sku(sku_id)
                if existing_sku:
                    print(f"   ℹ️  SKU already exists, using existing SKU (ID: {sku_id})")
                    # Note: Price is NOT set here - it should be set after images are added
                    return existing_sku
            except Exception as e:
                # If we can't get the SKU, log but don't raise - return minimal dict to allow continuation
                print(f"   ⚠️  SKU {sku_id} already exists but could not retrieve it: {e}")
                print(f"   ℹ️  Continuing with existing SKU ID: {sku_id}")
                # Return a minimal SKU dict with the ID so the workflow can continue
                return {"Id": sku_id, "Name": name, "ProductId": product_id}
        
        # Only raise for non-409 errors
        if response.status_code != 409:
            response.raise_for_status()
        else:
            # 409 without sku_id - return empty dict to allow continuation
            print(f"   ⚠️  SKU creation returned 409 Conflict but no sku_id provided")
            return {}
    
    def update_sku(
        self,
        sku_id: int,
        is_active: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update a SKU (e.g. set IsActive). Fetches the SKU via API, updates the field, then PUTs back.
        
        Args:
            sku_id: SKU ID
            is_active: Whether the SKU is active (Display on website)
            
        Returns:
            Updated SKU data or None if SKU not found or update failed
        """
        current_sku = self.get_sku(sku_id)
        if not current_sku:
            return None
        if is_active is not None:
            current_sku["IsActive"] = is_active
        endpoint = f"pvt/stockkeepingunit/{sku_id}"
        response = self._request("PUT", endpoint, data=current_sku)
        if response.status_code == 200:
            return response.json()
        response.raise_for_status()
        return None
    
    def set_sku_price(
        self,
        sku_id: int,
        price: float,
        list_price: Optional[float] = None
    ):
        """
        Set SKU pricing using VTEX Pricing API.
        
        Args:
            sku_id: SKU ID
            price: Base price (price from website - this is the actual price found on the website)
            list_price: List price (defaults to base price if not provided)
            
        Note:
            Uses VTEX Pricing API: PUT https://api.vtex.com/{account_name}/pricing/prices/{skuId}
            With markup=0, costPrice is set to the website price, which results in basePrice = costPrice
        """
        # Use VTEX Pricing API endpoint
        pricing_base_url = f"https://api.vtex.com/{self.account_name}"
        endpoint = f"/pricing/prices/{sku_id}"
        url = f"{pricing_base_url}{endpoint}"
        
        # With markup=0, basePrice = costPrice
        # So we set costPrice to the website price to get basePrice = website price
        data = {
            "markup": 0,  # Markup is always zero
            "costPrice": price  # Website price as costPrice (with markup=0, basePrice = costPrice = website price)
        }
        
        try:
            response = requests.put(
                url,
                json=data,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code in [200, 201, 204]:
                return response.json() if response.text else {"status": "success"}
            
            # Log errors but don't raise - allow continuation
            if response.status_code != 200:
                print(f"         ⚠️  Failed to set price for SKU {sku_id}. Status: {response.status_code}")
                print(f"         ⚠️  Response: {response.text[:200]}")
            
            response.raise_for_status()
            return {}
        except Exception as e:
            print(f"         ⚠️  Error setting price for SKU {sku_id}: {e}")
            raise
    
    def set_sku_inventory_all_warehouses(
        self,
        sku_id: int,
        quantity: int = 100
    ) -> Dict[str, Any]:
        """
        Set SKU inventory/stock to the specified quantity for ALL available warehouses.
        
        Args:
            sku_id: SKU ID
            quantity: Stock quantity to set for all warehouses (defaults to 100)
            
        Returns:
            Dictionary with results for each warehouse
        """
        results = {}
        warehouses = self.list_warehouses()
        
        if not warehouses:
            print(f"         ⚠️  No warehouses found, using default warehouse")
            # Fallback to default warehouse
            warehouse_id = os.getenv("VTEX_WAREHOUSE_ID", "1_1")
            result = self.set_sku_inventory(sku_id, warehouse_id=warehouse_id, quantity=quantity)
            results[warehouse_id] = result
            return results
        
        print(f"         📦 Setting inventory to {quantity} for {len(warehouses)} warehouse(s)...")
        for warehouse in warehouses:
            if isinstance(warehouse, dict):
                warehouse_id = warehouse.get("Id") or warehouse.get("id")
                warehouse_name = warehouse.get("Name") or warehouse.get("name", "Unknown")
                
                if warehouse_id:
                    try:
                        result = self.set_sku_inventory(
                            sku_id=sku_id,
                            warehouse_id=warehouse_id,
                            quantity=quantity
                        )
                        results[warehouse_id] = {
                            "warehouse_name": warehouse_name,
                            "quantity": quantity,
                            "success": True,
                            "result": result
                        }
                        print(f"           ✓ {warehouse_name}: {quantity}")
                    except Exception as e:
                        results[warehouse_id] = {
                            "warehouse_name": warehouse_name,
                            "quantity": quantity,
                            "success": False,
                            "error": str(e)
                        }
                        print(f"           ⚠️  {warehouse_name}: Failed - {e}")
        
        return results
    
    def list_warehouses(self) -> List[Dict[str, Any]]:
        """
        List all available warehouses.
        
        Returns:
            List of warehouse dictionaries with Id and Name
        """
        # VTEX Logistics API endpoint for warehouses
        logistics_base_url = (
            f"https://{self.account_name}.{self.environment}.com.br"
        )
        endpoint = "/api/logistics/pvt/configuration/warehouses"
        url = f"{logistics_base_url}{endpoint}"
        
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                warehouses = response.json()
                return warehouses if isinstance(warehouses, list) else []
            
            # Log errors but don't raise - allow continuation
            if response.status_code != 200:
                print(f"         ⚠️  Failed to list warehouses. Status: {response.status_code}")
                print(f"         ⚠️  Response: {response.text[:200]}")
            
            return []
        except Exception as e:
            print(f"         ⚠️  Error listing warehouses: {e}")
            return []
    
    def set_sku_inventory(
        self,
        sku_id: int,
        warehouse_id: Optional[str] = None,
        quantity: int = 0,
        unlimited_quantity: bool = False
    ) -> Dict[str, Any]:
        """
        Set SKU inventory/stock for a specific warehouse.
        
        Args:
            sku_id: SKU ID
            warehouse_id: Warehouse ID (if None, uses default warehouse from env or "1_1")
            quantity: Stock quantity
            unlimited_quantity: Whether to set unlimited quantity
            
        Returns:
            API response data
        """
        # Use warehouse_id from env or default to "1_1" (common default warehouse)
        warehouse_id = warehouse_id or os.getenv("VTEX_WAREHOUSE_ID", "1_1")
        
        # VTEX Logistics API endpoint for inventory
        # Note: This uses the Logistics API, not Catalog API
        logistics_base_url = (
            f"https://{self.account_name}.{self.environment}.com.br"
        )
        endpoint = f"/api/logistics/pvt/inventory/skus/{sku_id}/warehouses/{warehouse_id}"
        url = f"{logistics_base_url}{endpoint}"
        
        data = {
            "quantity": quantity,
            "unlimitedQuantity": unlimited_quantity
        }
        
        try:
            response = requests.put(
                url,
                json=data,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code in [200, 201, 204]:
                raw = response.json() if response.text else None
                # VTEX may return a boolean or other non-dict; always return a dict for callers
                if isinstance(raw, dict):
                    if "success" not in raw:
                        raw["success"] = True
                    return raw
                return {"success": True, "raw": raw}
            
            # Log errors but don't raise - allow continuation
            if response.status_code != 200:
                print(f"         ⚠️  Failed to set inventory for SKU {sku_id} in warehouse {warehouse_id}. Status: {response.status_code}")
                print(f"         ⚠️  Response: {response.text[:200]}")
            
            return {"success": False}
        except Exception as e:
            print(f"         ⚠️  Error setting inventory for SKU {sku_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def set_sku_inventory_all_warehouses(
        self,
        sku_id: int,
        quantity: int = 100
    ) -> Dict[str, Any]:
        """
        Set SKU inventory for all available warehouses.
        
        Args:
            sku_id: SKU ID
            quantity: Stock quantity to set for each warehouse (default: 100)
            
        Returns:
            Dictionary mapping warehouse name/id to result dict with "success" key
        """
        warehouses = self.list_warehouses()
        results = {}
        
        if not warehouses:
            print(f"         ⚠️  No warehouses found, using default warehouse")
            default_result = self.set_sku_inventory(sku_id, quantity=quantity)
            results["default"] = default_result if isinstance(default_result, dict) else {"success": True, "raw": default_result}
            return results
        
        print(f"         📦 Setting inventory to {quantity} for {len(warehouses)} warehouse(s)")
        
        for warehouse in warehouses:
            warehouse_id = warehouse.get("Id") or warehouse.get("id")
            warehouse_name = warehouse.get("Name") or warehouse.get("name") or str(warehouse_id)
            
            if not warehouse_id:
                continue
            
            try:
                result = self.set_sku_inventory(
                    sku_id=sku_id,
                    warehouse_id=str(warehouse_id),
                    quantity=quantity
                )
                # Ensure every value is a dict with "success" so callers can safely .get("success")
                if not isinstance(result, dict):
                    result = {"success": True, "raw": result}
                elif "success" not in result:
                    result["success"] = True
                results[warehouse_name] = result
                print(f"           ✓ Warehouse {warehouse_name}: {quantity}")
            except Exception as e:
                print(f"           ⚠️  Warehouse {warehouse_name}: Failed - {e}")
                results[warehouse_name] = {"success": False, "error": str(e)}
        
        return results
    
    # ========== IMAGE OPERATIONS ==========
    
    def upload_product_image(
        self,
        product_id: int,
        image_url: str,
        file_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Upload product image from URL."""
        # VTEX requires downloading the image and uploading via API
        import requests as req
        
        # Download image
        img_response = req.get(image_url, timeout=30)
        img_response.raise_for_status()
        
        # Upload to VTEX
        endpoint = f"pvt/products/{product_id}/images"
        files = {
            "file": (file_name or "image.jpg", img_response.content, "image/jpeg")
        }
        headers = {
            "X-VTEX-API-AppKey": self.app_key,
            "X-VTEX-API-AppToken": self.app_token
        }
        url = f"{self.base_url}/api/catalog/{endpoint}"
        response = requests.post(url, files=files, headers=headers, timeout=60)
        
        if response.status_code == 200:
            return response.json()
        response.raise_for_status()
    
    def associate_sku_image(
        self,
        sku_id: int,
        image_url: str,
        file_name: str,
        is_main: bool = False,
        label: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Associate an image with a SKU using the VTEX API.
        
        Endpoint: POST /api/catalog/pvt/stockkeepingunit/{skuId}/file
        
        Args:
            sku_id: SKU ID
            image_url: Raw GitHub URL or public image URL
            file_name: File name (e.g., "10010801_1")
            is_main: True for the first/main image, False for others
            label: Image label (e.g., SKU name or "Product Image")
            
        Returns:
            API response data
        """
        endpoint = f"pvt/stockkeepingunit/{sku_id}/file"
        data = {
            "url": image_url,
            "name": file_name,
            "isMain": is_main,
            "label": label or "Product Image"
        }
        
        response = self._request("POST", endpoint, data=data)
        
        if response.status_code in [200, 201]:
            return response.json() if response.text else {"status": "success"}
        
        # Handle 409 Conflict - image may already be associated
        if response.status_code == 409:
            print(f"         ℹ️  Image already associated with SKU {sku_id}, continuing...")
            return {"status": "already_exists", "sku_id": sku_id}
        
        # Log error but don't raise - allow continuation
        if response.status_code not in [200, 201, 409]:
            print(f"         ⚠️  Failed to associate image with SKU {sku_id}. Status: {response.status_code}")
            print(f"         ⚠️  Response: {response.text[:200]}")
        
        return {}

