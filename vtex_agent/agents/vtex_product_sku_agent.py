"""VTEX Product/SKU Agent - Creates products and SKUs in VTEX."""
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import re
import time

from ..clients.vtex_client import VTEXClient
from ..utils.state_manager import save_state, load_state, load_custom_prompt
from ..utils.logger import get_agent_logger
from ..utils.validation import extract_product_id, extract_sku_id, normalize_spec_name

if TYPE_CHECKING:
    from .vtex_category_tree_agent import VTEXCategoryTreeAgent


class VTEXProductSKUAgent:
    """Agent responsible for creating products and SKUs in VTEX."""
    
    def __init__(
        self,
        vtex_client: Optional[VTEXClient] = None,
        field_type_overrides: Optional[Dict[str, str]] = None,
        category_tree_agent: Optional["VTEXCategoryTreeAgent"] = None,
    ):
        self.logger = get_agent_logger("vtex_product_sku_agent")
        self.vtex_client = vtex_client or VTEXClient()
        self.category_tree_agent = category_tree_agent
        
        # Track created products
        self.products = {}
        
        # Load field type overrides from custom prompt or use provided overrides
        self.field_type_overrides = field_type_overrides or self._load_field_type_overrides()
        
        # Track dynamically created specification fields (not used - specifications disabled)
        self.created_spec_fields = {}
    
    def _load_field_type_overrides(self) -> Dict[str, str]:
        """
        Load field type overrides from state file.
        
        Looks for field_type_overrides.json in state directory.
        Format: {"Specification Name": "FieldType"} e.g., {"Material": "Combo", "Peso": "Number"}
        
        Returns:
            Dictionary mapping specification names to field types
        """
        overrides = {}
        
        # Try loading from dedicated state file
        try:
            state = load_state("field_type_overrides")
            if state and isinstance(state, dict):
                overrides.update(state)
                self.logger.info(f"Loaded {len(overrides)} field type overrides from state")
        except Exception as e:
            self.logger.debug(f"Could not load field type overrides from state: {e}")
        
        # Also check custom prompt for field type instructions
        # Format: "Field Type Overrides: Material=Combo, Peso=Number, Acabamento=Combo"
        try:
            custom_prompt = load_custom_prompt()
            if custom_prompt and "field type" in custom_prompt.lower():
                # Simple parsing: look for "Field Type Overrides:" or similar
                import re
                # Match patterns like "Material=Combo" or "Material: Combo"
                matches = re.findall(r'(\w+)\s*[=:]\s*(\w+)', custom_prompt, re.IGNORECASE)
                for spec_name, field_type in matches:
                    overrides[spec_name] = field_type.capitalize()  # Normalize to Title case
                
                if matches:
                    self.logger.info(f"Loaded {len(matches)} field type overrides from custom prompt")
        except Exception as e:
            self.logger.debug(f"Could not parse field type overrides from custom prompt: {e}")
        
        if overrides:
            self.logger.info(f"Total field type overrides loaded: {overrides}")
        
        return overrides
    
    def create_products_and_skus(
        self,
        legacy_site_data: Dict[str, Any],
        vtex_category_tree: Dict[str, Any],
        vtex_specifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create products and SKUs in VTEX.
        
        Args:
            legacy_site_data: Output from Legacy Site Agent
            vtex_category_tree: Output from VTEX Category Tree Agent
            vtex_specifications: Dictionary with specification fields (can be empty or loaded from state)
            
        Returns:
            Dictionary with created products and SKUs
        """
        self.logger.info("Starting product and SKU creation")
        
        # Try to load from state
        state = load_state("vtex_products_skus")
        if state and state.get("products"):
            self.logger.info("Loaded products from state")
            self.products = state.get("products", {})
            return self._format_output()
        
        products = legacy_site_data.get("products", [])
        categories = vtex_category_tree.get("categories", {})
        departments = vtex_category_tree.get("departments", {})
        brands = vtex_category_tree.get("brands", {})
        spec_fields = vtex_specifications.get("specification_fields", {})
        
        # Note: Specification fields will be created on-the-fly if they don't exist
        if not spec_fields:
            self.logger.info(
                "No specification fields found in state. "
                "Specification fields will be created automatically as needed during product creation."
            )
        
        # Helper functions to get IDs from category tree
        def get_category_id_for_product(product: Dict[str, Any]) -> Optional[int]:
            """Get category ID for a product."""
            categories_list = product.get("categories", [])
            if not categories_list:
                category = product.get("category", {})
                if category:
                    categories_list = [category]
            if not categories_list:
                return None
            
            # Skip only obvious root-level categories that aren't real departments
            # NOTE: We intentionally do NOT skip "Início"/"Inicio" here because many
            # sites use it as the actual department in VTEX.
            skip_names = {"home", "root", "default"}
            
            # Find the department (usually level 2, but could be level 1)
            dept_name = None
            dept_index = 0
            
            for i, cat_info in enumerate(categories_list):
                cat_name = cat_info.get("Name", "").strip()
                cat_name_lower = cat_name.lower()
                
                # Skip root categories
                if cat_name_lower in skip_names:
                    continue
                
                # Try to find this as a department (case-insensitive)
                for dept_key, dept_data in departments.items():
                    if dept_key.lower() == cat_name_lower or dept_data.get("name", "").lower() == cat_name_lower:
                        dept_name = dept_key
                        dept_index = i
                        break
                
                if dept_name:
                    break
            
            if not dept_name:
                # If no department found, try first non-skip category
                for cat_info in categories_list:
                    cat_name = cat_info.get("Name", "").strip()
                    if cat_name.lower() not in skip_names:
                        # Try case-insensitive match
                        for dept_key, dept_data in departments.items():
                            if dept_key.lower() == cat_name.lower() or dept_data.get("name", "").lower() == cat_name.lower():
                                dept_name = dept_key
                                break
                        if dept_name:
                            break
            
            if not dept_name or dept_name not in departments:
                # Fallback: product categories may not start with department name (e.g. "Linhas"
                # under department "Início"). Try each department as root and match full path.
                for _dept_key, dept_data in departments.items():
                    parent_id = dept_data["id"]
                    matched_any = False
                    matched_all = True
                    for cat_info in categories_list:
                        cat_name = cat_info.get("Name", "").strip()
                        if not cat_name or cat_name.lower() in skip_names:
                            continue
                        found = False
                        for _cat_key, cat_data in categories.items():
                            if (
                                cat_data.get("name", "").strip().lower() == cat_name.lower()
                                and cat_data.get("parent_id") == parent_id
                            ):
                                parent_id = cat_data.get("id")
                                found = True
                                matched_any = True
                                break
                        if not found:
                            matched_all = False
                            break
                    if matched_all and matched_any:
                        return parent_id
                return None

            parent_id = departments[dept_name]["id"]

            # If only department level, return it
            if len(categories_list) <= dept_index + 1:
                return parent_id

            # Traverse remaining categories
            for cat_info in categories_list[dept_index + 1:]:
                cat_name = cat_info.get("Name", "").strip()
                if not cat_name or cat_name.lower() in skip_names:
                    continue

                # Try to find category with matching parent (case-insensitive)
                found = False
                for cat_key, cat_data in categories.items():
                    # Check if name matches (case-insensitive) and parent matches
                    cat_data_name = cat_data.get("name", "").strip()
                    cat_data_parent = cat_data.get("parent_id")

                    if (
                        cat_data_name.lower() == cat_name.lower()
                        and cat_data_parent == parent_id
                    ):
                        parent_id = cat_data.get("id")
                        found = True
                        break

                if not found:
                    # If exact match not found, continue with current parent_id
                    break

            return parent_id
        
        def get_brand_id(brand_name: str) -> Optional[int]:
            """Get brand ID by name (case-insensitive)."""
            if not brand_name:
                return None
            
            target = brand_name.strip().lower()
            
            # Try matching against both the brand dict keys and their stored names
            for brand_key, brand_data in brands.items():
                key_name = str(brand_key).strip().lower()
                data_name = str(brand_data.get("name", "")).strip().lower()
                
                if target == key_name or target == data_name:
                    return brand_data.get("id")
            
            return None
        
        def get_spec_field_id(category_id: int, spec_name: str) -> Optional[int]:
            """Get specification field ID."""
            normalized = normalize_spec_name(spec_name)
            spec_key = f"{category_id}::{normalized}"
            field_data = spec_fields.get(spec_key)
            if field_data:
                return field_data.get("id")
            return None
        
        print(f"\n📦 Processing {len(products)} products...")
        
        for i, product_data in enumerate(products, 1):
            print(f"\n   [{i}/{len(products)}] Processing product...")
            self.logger.info(f"Processing product {i}/{len(products)}")
            
            try:
                # Get category ID
                category_id = get_category_id_for_product(product_data)
                if not category_id:
                    product_categories = product_data.get("categories", [])
                    category_names = [c.get("Name", "") for c in product_categories]
                    available_depts = list(departments.keys())
                    self.logger.warning(
                        f"Could not determine category ID for product. "
                        f"Product categories: {category_names}. "
                        f"Available departments: {available_depts}. Skipping product."
                    )
                    continue
                
                # Get brand ID
                brand_name = product_data.get("brand", {}).get("Name", "Default")
                brand_id = get_brand_id(brand_name)
                if not brand_id:
                    self.logger.warning(f"Could not determine brand ID for {brand_name}, skipping")
                    continue
                
                # Create product
                product_info = product_data.get("product", {})
                product_name = product_info.get("Name", "Product")
                print(f"     📦 Creating product: {product_name}")
                
                # Extract product ID
                extracted_product_id = product_info.get("ProductId")
                product_id_param = extract_product_id(extracted_product_id)
                
                product = self.vtex_client.create_product(
                    name=product_name,
                    category_id=category_id,
                    brand_id=brand_id,
                    description=product_info.get("Description"),
                    short_description=product_info.get("ShortDescription"),
                    is_active=True,  # Always set Display on website flag active
                    is_visible=True,  # Always set product visible when creating
                    show_without_stock=product_info.get("ShowWithoutStock", True),
                    product_id=product_id_param
                )
                
                product_id = product.get("Id") if isinstance(product, dict) else None
                if not product_id:
                    self.logger.warning(f"Could not get product ID, skipping")
                    continue
                
                # Ensure IsActive is set to True (in case product already existed)
                try:
                    if not product.get("IsActive", False) or not product.get("IsVisible", False):
                        self.vtex_client.update_product(product_id, is_active=True, is_visible=True)
                        if not product.get("IsActive", False):
                            print(f"       ✓ Updated product IsActive flag to True")
                        if not product.get("IsVisible", False):
                            print(f"       ✓ Updated product IsVisible flag to True")
                except Exception as update_error:
                    self.logger.warning(f"Could not update product flags for product {product_id}: {update_error}")
                
                if extracted_product_id:
                    print(f"       ℹ️  Extracted Product ID: {extracted_product_id}")
                
                # Set specifications
                specifications = product_data.get("specifications", [])
                if specifications:
                    print(f"     📋 Processing {len(specifications)} specifications...")
                    self._set_product_specifications(
                        product_id,
                        category_id,
                        specifications,
                        spec_fields,
                        category_tree=vtex_category_tree
                    )
                
                # Create SKUs
                skus = product_data.get("skus", [])
                if not skus:
                    # Create default SKU
                    skus = [{
                        "Name": "Default",
                        "EAN": f"EAN{product_id}",
                        "IsActive": True
                    }]
                
                created_skus = []
                for sku_data in skus:
                    sku_name = sku_data.get("Name", "Default")
                    print(f"       🔢 Creating SKU: {sku_name}")
                    
                    extracted_sku_id = sku_data.get("SkuId")
                    sku_id_param = extract_sku_id(extracted_sku_id)
                    
                    if extracted_sku_id:
                        print(f"         ℹ️  Extracted SKU ID: {extracted_sku_id}")
                    
                    sku = self.vtex_client.create_sku(
                        product_id=product_id,
                        name=sku_name,
                        ean=sku_data.get("EAN", f"EAN{product_id}"),
                        is_active=False,  # VTEX requires files/components before SKU can be active
                        ref_id=sku_data.get("RefId") or extracted_sku_id,
                        price=sku_data.get("Price") or 0,  # Ensure price is set (default to 0)
                        list_price=sku_data.get("ListPrice") or sku_data.get("Price") or 0,
                        package_height=1,  # Set packaged dimensions to 1
                        package_width=1,
                        package_length=1,
                        package_weight=1,  # Set packaged weight to 1
                        height=1,  # Set unpackaged dimensions to 1
                        width=1,
                        length=1,
                        weight=1,  # Set unpackaged weight to 1
                        sku_id=sku_id_param
                    )
                    
                    sku_id = sku.get("Id") if isinstance(sku, dict) else None
                    if sku_id:
                        # Note: SKU activation is done after images in the flow that uses this (if any).
                        # Note: Price and inventory are NOT set here
                        # They should be set after images are added in the correct order:
                        # Create SKU > Add images > Add price > Add inventory
                        
                        created_skus.append({
                            "id": sku_id,
                            "name": sku_name,
                            "sku_id_preserved": extracted_sku_id,
                            "ref_id": sku_data.get("RefId") or extracted_sku_id,
                            "created": True
                        })
                
                # Store product
                product_url = product_data.get("url", f"product_{product_id}")
                self.products[product_url] = {
                    "id": product_id,
                    "name": product_name,
                    "category_id": category_id,
                    "brand_id": brand_id,
                    "product_id_preserved": extracted_product_id,
                    "created": True,
                    "skus": created_skus,
                    "specifications_set": len(specifications)
                }
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                self.logger.error(f"Error processing product: {e}", exc_info=True)
                print(f"     ⚠️  Error processing product: {e}")
                continue
        
        # Save output
        output = self._format_output()
        save_state("vtex_products_skus", output)
        
        # Also update specification fields state with dynamically created fields
        if self.created_spec_fields:
            spec_state = load_state("vtex_specifications") or {}
            existing_fields = spec_state.get("specification_fields", {})
            existing_fields.update(self.created_spec_fields)
            spec_state["specification_fields"] = existing_fields
            save_state("vtex_specifications", spec_state)
            self.logger.info(f"Updated specification fields state with {len(self.created_spec_fields)} dynamically created fields")
        
        self.logger.info(f"Product/SKU creation complete. Created {len(self.products)} products")
        
        return output
    
    def _create_specification_field_if_missing(
        self,
        spec_name: str,
        category_id: int,
        sample_value: Optional[str] = None
    ) -> Optional[int]:
        """
        Create a specification field if it doesn't exist.
        Specifications are disabled - this method always returns None.
        
        Args:
            spec_name: Specification name
            category_id: Category ID
            sample_value: Sample value to help determine field type
            
        Returns:
            None (specifications are disabled)
        """
        normalized = normalize_spec_name(spec_name)
        self.logger.info(f"Specifications disabled - skipping creation of specification field '{normalized}'")
        return None
    
    def _set_product_specifications(
        self,
        product_id: int,
        category_id: int,
        specifications: List[Dict[str, Any]],
        spec_fields: Dict[str, Dict[str, Any]],
        category_tree: Optional[Dict[str, Any]] = None
    ):
        """
        Set specification values on a product.
        Specifications are disabled - this method does nothing.
        
        Args:
            product_id: Product ID
            category_id: Category ID for the product
            specifications: List of specification dicts with Name and Value
            spec_fields: Dictionary of specification fields from state
            category_tree: Optional category tree to check parent categories
        """
        # Specifications are disabled - skip all specification operations
        self.logger.info(f"Specifications disabled - skipping specification setting for product {product_id}")
        return
        # Build category hierarchy if available
        category_hierarchy = {}
        if category_tree:
            categories = category_tree.get("categories", {})
            departments = category_tree.get("departments", {})
            
            # Build hierarchy map
            for dept_name, dept_data in departments.items():
                dept_id = dept_data.get("id")
                if dept_id:
                    category_hierarchy[dept_id] = {"parent_id": None}
            
            for cat_key, cat_data in categories.items():
                cat_id = cat_data.get("id")
                parent_id = cat_data.get("parent_id")
                if cat_id:
                    category_hierarchy[cat_id] = {"parent_id": parent_id}
        
        for spec in specifications:
            spec_name = spec.get("Name", "")
            spec_value = spec.get("Value", "")
            
            if not spec_name or not spec_value:
                continue
            
            # Get field ID - try current category first, then parent categories
            normalized = normalize_spec_name(spec_name)
            field_id = None
            field_data = None
            
            # Try current category
            spec_key = f"{category_id}::{normalized}"
            field_data = spec_fields.get(spec_key)
            
            # If not found, try parent categories (specs can be inherited)
            if not field_data and category_hierarchy:
                current_cat_id = category_id
                max_depth = 5  # Prevent infinite loops
                depth = 0
                
                while not field_data and current_cat_id and depth < max_depth:
                    parent_info = category_hierarchy.get(current_cat_id)
                    if not parent_info:
                        break
                    
                    parent_id = parent_info.get("parent_id")
                    if parent_id:
                        parent_key = f"{parent_id}::{normalized}"
                        field_data = spec_fields.get(parent_key)
                        if field_data:
                            self.logger.debug(
                                f"Found specification '{spec_name}' in parent category {parent_id} "
                                f"(product category: {category_id})"
                            )
                            break
                        current_cat_id = parent_id
                    else:
                        break
                    depth += 1
            
            # If still not found in state, try querying VTEX API directly
            if not field_data:
                try:
                    existing_fields = self.vtex_client.list_specification_fields(category_id)
                    for field in existing_fields:
                        if isinstance(field, dict) and field.get("Name") == normalized:
                            field_id = field.get("Id")
                            self.logger.info(
                                f"Found specification field '{spec_name}' (ID: {field_id}) "
                                f"via VTEX API for category {category_id}"
                            )
                            break
                except Exception as e:
                    self.logger.debug(f"Could not query VTEX for fields in category {category_id}: {e}")
            
            # Get field ID from found data
            if field_data:
                field_id = field_data.get("id")
            
            # If still not found, create the field automatically (defaults to Text type)
            if not field_id:
                self.logger.info(
                    f"Specification field '{spec_name}' not found. "
                    f"Creating automatically as Text type (unless overridden in prompt)."
                )
                field_id = self._create_specification_field_if_missing(
                    spec_name=spec_name,
                    category_id=category_id,
                    sample_value=spec_value
                )
                
                if not field_id:
                    self.logger.warning(
                        f"Could not create or find field ID for specification: {spec_name} "
                        f"(normalized: {normalized}, category: {category_id}). "
                        f"Skipping this specification."
                    )
                    continue
            
            # Get field type if available
            field_type = "Text"  # Default
            if field_data:
                field_type = field_data.get("field_type", "Text")
            elif spec_key in self.created_spec_fields:
                field_type = self.created_spec_fields[spec_key].get("field_type", "Text")
            
            # Set specification value
            try:
                result = self.vtex_client.set_product_specification(
                    product_id=product_id,
                    field_id=field_id,
                    field_value=spec_value,
                    field_type=field_type
                )
                if result:
                    self.logger.debug(f"✅ Set specification '{spec_name}' = '{spec_value}' for product {product_id}")
                    print(f"         ✓ Set {spec_name}: {spec_value}")
                else:
                    self.logger.warning(f"⚠️  Setting specification '{spec_name}' returned empty result")
            except Exception as e:
                self.logger.error(f"❌ Error setting specification {spec_name}: {e}")
                print(f"         ✗ Failed to set {spec_name}: {str(e)[:50]}")
    
    def create_single_product(
        self,
        product_data: Dict[str, Any],
        vtex_category_tree: Dict[str, Any],
        vtex_specifications: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Create a single product in VTEX and set its specifications.
        
        Args:
            product_data: Product data from legacy site
            vtex_category_tree: Output from VTEX Category Tree Agent
            vtex_specifications: Dictionary with specification fields (can be empty; fields will be created on-the-fly if needed)
            
        Returns:
            Dictionary with product info including product_id, or None if failed
        """
        categories = vtex_category_tree.get("categories", {})
        departments = vtex_category_tree.get("departments", {})
        brands = vtex_category_tree.get("brands", {})
        spec_fields = vtex_specifications.get("specification_fields", {})
        
        # Helper function to get category ID
        def get_category_id_for_product(product: Dict[str, Any]) -> Optional[int]:
            """Get category ID for a product."""
            categories_list = product.get("categories", [])
            if not categories_list:
                category = product.get("category", {})
                if category:
                    categories_list = [category]
            if not categories_list:
                return None
            
            # Same logic as in create_products_and_skus: do not skip "Início"/"Inicio"
            # so products whose hierarchy starts with that name can still resolve.
            skip_names = {"home", "root", "default"}
            dept_name = None
            dept_index = 0
            
            for i, cat_info in enumerate(categories_list):
                cat_name = cat_info.get("Name", "").strip()
                cat_name_lower = cat_name.lower()
                
                if cat_name_lower in skip_names:
                    continue
                
                for dept_key, dept_data in departments.items():
                    if dept_key.lower() == cat_name_lower or dept_data.get("name", "").lower() == cat_name_lower:
                        dept_name = dept_key
                        dept_index = i
                        break
                
                if dept_name:
                    break
            
            if not dept_name:
                for cat_info in categories_list:
                    cat_name = cat_info.get("Name", "").strip()
                    if cat_name.lower() not in skip_names:
                        for dept_key, dept_data in departments.items():
                            if dept_key.lower() == cat_name.lower() or dept_data.get("name", "").lower() == cat_name.lower():
                                dept_name = dept_key
                                break
                        if dept_name:
                            break
            
            if not dept_name or dept_name not in departments:
                # Fallback: product categories may not start with department name (e.g. "Linhas"
                # under department "Início"). Try each department as root and match full path.
                for _dept_key, dept_data in departments.items():
                    parent_id = dept_data["id"]
                    matched_any = False
                    matched_all = True
                    for cat_info in categories_list:
                        cat_name = cat_info.get("Name", "").strip()
                        if not cat_name or cat_name.lower() in skip_names:
                            continue
                        found = False
                        for _cat_key, cat_data in categories.items():
                            if (
                                cat_data.get("name", "").strip().lower() == cat_name.lower()
                                and cat_data.get("parent_id") == parent_id
                            ):
                                parent_id = cat_data.get("id")
                                found = True
                                matched_any = True
                                break
                        if not found:
                            matched_all = False
                            break
                    if matched_all and matched_any:
                        return parent_id
                return None

            parent_id = departments[dept_name]["id"]

            if len(categories_list) <= dept_index + 1:
                return parent_id

            for cat_info in categories_list[dept_index + 1:]:
                cat_name = cat_info.get("Name", "").strip()
                if not cat_name or cat_name.lower() in skip_names:
                    continue

                found = False
                for cat_key, cat_data in categories.items():
                    cat_data_name = cat_data.get("name", "").strip()
                    cat_data_parent = cat_data.get("parent_id")

                    if (
                        cat_data_name.lower() == cat_name.lower()
                        and cat_data_parent == parent_id
                    ):
                        parent_id = cat_data.get("id")
                        found = True
                        break

                if not found:
                    break

            return parent_id
        
        def get_brand_id(brand_name: str) -> Optional[int]:
            """Get brand ID by name (case-insensitive)."""
            if not brand_name:
                return None
            
            target = brand_name.strip().lower()
            
            for brand_key, brand_data in brands.items():
                key_name = str(brand_key).strip().lower()
                data_name = str(brand_data.get("name", "")).strip().lower()
                
                if target == key_name or target == data_name:
                    return brand_data.get("id")
            
            return None
        
        # Get category ID; if missing, ask category tree agent to create/find the path
        category_id = get_category_id_for_product(product_data)
        updated_tree_from_ensure = None
        if not category_id and self.category_tree_agent:
            category_id, updated_tree_from_ensure = self.category_tree_agent.ensure_category_for_product(
                product_data
            )
            if updated_tree_from_ensure:
                # Use updated tree for rest of this call (brands etc. unchanged; categories/departments updated)
                categories = updated_tree_from_ensure.get("categories", {})
                departments = updated_tree_from_ensure.get("departments", {})
        if not category_id:
            product_categories = product_data.get("categories", [])
            category_names = [c.get("Name", "") for c in product_categories]
            available_depts = list(departments.keys())
            self.logger.warning(
                f"Could not determine category ID for product. "
                f"Product categories: {category_names}. "
                f"Available departments: {available_depts}. Skipping product."
            )
            return None

        # Get brand ID (case-insensitive)
        brand_name = product_data.get("brand", {}).get("Name", "Default")
        brand_id = get_brand_id(brand_name)
        if not brand_id:
            self.logger.warning(f"Could not determine brand ID for {brand_name}, skipping")
            return None
        
        # Create product
        product_info = product_data.get("product", {})
        product_name = product_info.get("Name", "Product")
        print(f"     📦 Creating product: {product_name}")
        
        extracted_product_id = product_info.get("ProductId")
        product_id_param = extract_product_id(extracted_product_id)
        
        product_id = None
        try:
            product = self.vtex_client.create_product(
                name=product_name,
                category_id=category_id,
                brand_id=brand_id,
                description=product_info.get("Description"),
                short_description=product_info.get("ShortDescription"),
                is_active=True,  # Always set Display on website flag active
                is_visible=True,  # Always set product visible when creating
                show_without_stock=product_info.get("ShowWithoutStock", True),
                product_id=product_id_param
            )
            
            product_id = product.get("Id") if isinstance(product, dict) else None
            if product_id:
                print(f"       ✅ Product created with ID: {product_id}")
                # Ensure IsActive is set to True (in case product already existed)
                try:
                    if not product.get("IsActive", False) or not product.get("IsVisible", False):
                        self.vtex_client.update_product(product_id, is_active=True, is_visible=True)
                        if not product.get("IsActive", False):
                            print(f"       ✓ Updated product IsActive flag to True")
                        if not product.get("IsVisible", False):
                            print(f"       ✓ Updated product IsVisible flag to True")
                except Exception as update_error:
                    self.logger.warning(f"Could not update product flags for product {product_id}: {update_error}")
        except Exception as e:
            # Handle case where product already exists (409 Conflict)
            if "409" in str(e) or "Conflict" in str(e):
                if product_id_param is not None:
                    print(f"       ℹ️  Product already exists, retrieving existing product (ID: {product_id_param})...")
                    try:
                        product = self.vtex_client.get_product(product_id_param)
                        if product:
                            product_id = product.get("Id") if isinstance(product, dict) else product_id_param
                            print(f"       ✅ Using existing product with ID: {product_id}")
                        else:
                            self.logger.warning(f"Product {product_id_param} already exists but could not retrieve it")
                            # Use the provided product ID to continue
                            product_id = product_id_param
                            product = {"Id": product_id, "Name": product_info.get("Name", "Existing Product")}
                            print(f"       ℹ️  Continuing with existing product ID: {product_id}")
                    except Exception as get_error:
                        self.logger.error(f"Error retrieving existing product {product_id_param}: {get_error}")
                        # Use the provided product ID to continue
                        product_id = product_id_param
                        product = {"Id": product_id, "Name": product_info.get("Name", "Existing Product")}
                        print(f"       ℹ️  Continuing with existing product ID: {product_id}")
                else:
                    self.logger.error(f"Product creation failed with 409 but no product_id provided: {e}")
                    return None
            else:
                # Re-raise if it's a different error
                self.logger.error(f"Error creating product: {e}")
                raise
        
        if not product_id:
            self.logger.warning(f"Could not get product ID, skipping")
            return None
        
        if extracted_product_id:
            print(f"       ℹ️  Extracted Product ID: {extracted_product_id}")
        
        # Use updated tree when category was ensured so specs use correct category tree
        effective_category_tree = updated_tree_from_ensure if updated_tree_from_ensure else vtex_category_tree
        
        # Set specifications
        specifications = product_data.get("specifications", [])
        if specifications:
            print(f"     📋 Processing {len(specifications)} specifications...")
            self._set_product_specifications(
                product_id,
                category_id,
                specifications,
                spec_fields,
                category_tree=effective_category_tree
            )
        
        # Store product info
        product_url = product_data.get("url", f"product_{product_id}")
        product_info_dict = {
            "id": product_id,
            "name": product_name,
            "category_id": category_id,
            "brand_id": brand_id,
            "product_id_preserved": extracted_product_id,
            "created": True,
            "skus": [],
            "specifications_set": len(specifications)
        }
        if updated_tree_from_ensure is not None:
            product_info_dict["vtex_category_tree"] = updated_tree_from_ensure
        
        self.products[product_url] = product_info_dict
        
        time.sleep(0.5)  # Rate limiting
        
        return product_info_dict
    
    def create_single_sku(
        self,
        product_id: int,
        product_url: str,
        sku_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Create a single SKU for a product.
        
        Args:
            product_id: VTEX product ID
            product_url: Product URL (key in self.products)
            sku_data: SKU data from legacy site
            
        Returns:
            Dictionary with SKU info including sku_id, or None if failed
        """
        sku_name = sku_data.get("Name", "Default")
        print(f"       🔢 Creating SKU: {sku_name}")
        
        extracted_sku_id = sku_data.get("SkuId")
        sku_id_param = extract_sku_id(extracted_sku_id)
        
        if extracted_sku_id:
            print(f"         ℹ️  Extracted SKU ID: {extracted_sku_id}")
        
        try:
            sku = self.vtex_client.create_sku(
                product_id=product_id,
                name=sku_name,
                ean=sku_data.get("EAN", f"EAN{product_id}"),
                is_active=False,  # VTEX requires files/components before SKU can be active
                ref_id=sku_data.get("RefId") or extracted_sku_id,
                price=sku_data.get("Price") or 0,  # Ensure price is set (default to 0)
                list_price=sku_data.get("ListPrice") or sku_data.get("Price") or 0,
                package_height=1,  # Set packaged dimensions to 1
                package_width=1,
                package_length=1,
                package_weight=1,  # Set packaged weight to 1
                height=1,  # Set unpackaged dimensions to 1
                width=1,
                length=1,
                weight=1,  # Set unpackaged weight to 1
                sku_id=sku_id_param
            )
        except Exception as e:
            # Handle case where SKU already exists (409 Conflict)
            if "409" in str(e) or "Conflict" in str(e):
                if sku_id_param is not None:
                    print(f"       ℹ️  SKU already exists, retrieving existing SKU (ID: {sku_id_param})...")
                    try:
                        sku = self.vtex_client.get_sku(sku_id_param)
                        if sku:
                            sku_id = sku.get("Id") if isinstance(sku, dict) else sku_id_param
                            print(f"       ✅ Using existing SKU with ID: {sku_id}")
                        else:
                            self.logger.warning(f"SKU {sku_id_param} already exists but could not retrieve it")
                            # Use the provided SKU ID to continue
                            sku_id = sku_id_param
                            sku = {"Id": sku_id, "Name": sku_name, "ProductId": product_id}
                    except Exception as get_error:
                        self.logger.error(f"Error retrieving existing SKU {sku_id_param}: {get_error}")
                        # Use the provided SKU ID to continue
                        sku_id = sku_id_param
                        sku = {"Id": sku_id, "Name": sku_name, "ProductId": product_id}
                    
                    # Note: Price and inventory are NOT set here for existing SKUs
                    # They should be set after images are added in the correct order
                else:
                    self.logger.error(f"SKU creation failed with 409 but no sku_id provided: {e}")
                    return None
            else:
                # Re-raise if it's a different error
                self.logger.error(f"Error creating SKU: {e}")
                raise
        
        sku_id = sku.get("Id") if isinstance(sku, dict) else None
        if not sku_id:
            self.logger.warning(f"Could not get SKU ID for {sku_name}")
            return None
        
        # Note: SKU activation (IsActive=true) is done after images are associated (e.g. in migration_agent).
        # Note: Price and inventory are NOT set here
        # They should be set after images are added in the correct order:
        # Create SKU > Add images > Add price > Add inventory
        
        # Store SKU info
        sku_info = {
            "id": sku_id,
            "name": sku_name,
            "sku_id_preserved": extracted_sku_id,
            "ref_id": sku_data.get("RefId") or extracted_sku_id,
            "created": True
        }
        
        # Add to product's SKU list
        if product_url in self.products:
            self.products[product_url]["skus"].append(sku_info)
        
        return sku_info
    
    def set_sku_price_and_inventory(
        self,
        sku_id: int,
        sku_data: Dict[str, Any]
    ) -> bool:
        """
        Set price and inventory for a SKU.
        This should be called after images are added.
        
        Args:
            sku_id: SKU ID
            sku_data: SKU data dictionary with Price, ListPrice, and Inventory fields
            
        Returns:
            True if both price and inventory were set successfully, False otherwise
        """
        success = True
        
        # Set price
        try:
            price_value = sku_data.get("Price") or 0
            list_price_value = sku_data.get("ListPrice") or price_value
            self.vtex_client.set_sku_price(sku_id, price_value, list_price_value)
            print(f"         ✓ Price set: {price_value}")
        except Exception as price_error:
            self.logger.warning(f"Could not set price for SKU {sku_id}: {price_error}")
            success = False
        
        # Set inventory
        try:
            inventory_quantity = sku_data.get("Inventory", 0)  # Default to 0 if not specified
            self.vtex_client.set_sku_inventory(sku_id, quantity=inventory_quantity)
            print(f"         ✓ Inventory set: {inventory_quantity}")
        except Exception as inventory_error:
            self.logger.warning(f"Could not set inventory for SKU {sku_id}: {inventory_error}")
            success = False
        
        return success
    
    def _format_output(self) -> Dict[str, Any]:
        """Format output JSON."""
        total_skus = sum(len(p.get("skus", [])) for p in self.products.values())
        
        return {
            "products": self.products,
            "dynamically_created_spec_fields": self.created_spec_fields,
            "summary": {
                "total_products": len(self.products),
                "total_skus": total_skus,
                "products_created": sum(1 for p in self.products.values() if p.get("created")),
                "skus_created": total_skus,
                "spec_fields_created_dynamically": sum(1 for f in self.created_spec_fields.values() if f.get("created"))
            }
        }

