"""VTEX Category Tree Agent - Creates and manages VTEX category hierarchy."""
from typing import Dict, Any, List, Optional, Tuple
import time

from ..clients.vtex_client import VTEXClient
from ..utils.state_manager import save_state, load_state
from ..utils.logger import get_agent_logger
from ..utils.validation import normalize_category_name, normalize_brand_name
from ..utils.error_handler import retry_with_exponential_backoff


class VTEXCategoryTreeAgent:
    """Agent responsible for creating VTEX category tree and brands."""
    
    def __init__(self, vtex_client: Optional[VTEXClient] = None):
        self.logger = get_agent_logger("vtex_category_tree_agent")
        self.vtex_client = vtex_client or VTEXClient()
        
        # Track created entities
        self.departments = {}
        self.categories = {}
        self.brands = {}
    
    def create_category_tree(self, legacy_site_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create VTEX category tree from legacy site data.
        
        Args:
            legacy_site_data: Output from Legacy Site Agent
            
        Returns:
            Dictionary with VTEX category structure (departments, categories, brands)
        """
        self.logger.info("Starting category tree creation")
        
        # Load from state if present (warm start), but always process current products
        # so that new categories from this import are created
        state = load_state("vtex_category_tree")
        if state and state.get("departments"):
            self.logger.info("Loaded category tree from state; will extend with current products")
            self.departments = state.get("departments", {})
            self.categories = state.get("categories", {})
            self.brands = state.get("brands", {})
        
        products = legacy_site_data.get("products", [])
        self.logger.info(f"Processing {len(products)} products for category tree")
        
        # Evaluate existing VTEX structure (avoid creating duplicates)
        existing_categories = self._evaluate_existing_categories()
        existing_brands = self._evaluate_existing_brands()
        
        # Process all products to build/extend category tree (creates missing depts/cats/brands)
        for product in products:
            self._process_product_categories(product, existing_categories)
            self._process_product_brand(product, existing_brands)
        
        # Save output
        output = self._format_output()
        save_state("vtex_category_tree", output)
        
        self.logger.info(f"Category tree creation complete. Created {len(self.departments)} departments, {len(self.categories)} categories, {len(self.brands)} brands")
        
        return output
    
    def _evaluate_existing_categories(self) -> Dict[str, Dict[str, Any]]:
        """Evaluate existing categories in VTEX (name -> cat). Use for backward compatibility."""
        self.logger.info("Evaluating existing VTEX categories")
        existing = {}
        try:
            categories = self.vtex_client.list_categories()
            for cat in categories:
                if isinstance(cat, dict):
                    name = cat.get("Name", "")
                    if name:
                        normalized = normalize_category_name(name)
                        existing[normalized] = cat
        except Exception as e:
            self.logger.warning(f"Error evaluating existing categories: {e}")
        self.logger.info(f"Found {len(existing)} existing categories")
        return existing

    def _existing_categories_by_parent(self) -> Dict[tuple, Dict[str, Any]]:
        """Return existing categories keyed by (parent_id int, normalized_name) so we only reuse under correct parent."""
        by_parent = {}
        try:
            categories = self.vtex_client.list_categories()
            for cat in categories:
                if not isinstance(cat, dict):
                    continue
                name = cat.get("Name", "") or cat.get("name", "")
                if not name:
                    continue
                raw = cat.get("FatherCategoryId") or cat.get("FatherCategoryID") or 0
                try:
                    parent_id = int(raw) if raw not in (None, "") else 0
                except (TypeError, ValueError):
                    parent_id = 0
                normalized = normalize_category_name(name)
                by_parent[(parent_id, normalized)] = cat
        except Exception as e:
            self.logger.warning(f"Error building categories by parent: {e}")
        return by_parent

    def _sync_tree_from_vtex(self) -> None:
        """
        Merge VTEX category tree into self.departments and self.categories.
        Only merges in categories from the API; does not clear existing tree if API returns empty or wrong shape.
        """
        try:
            categories = self.vtex_client.list_categories()
        except Exception as e:
            self.logger.warning(f"Could not list categories from VTEX for sync: {e}")
            return
        if not isinstance(categories, list) or not categories:
            return
        new_departments = {}
        new_categories = {}
        for cat in categories:
            if not isinstance(cat, dict):
                continue
            cat_id = cat.get("Id")
            name = cat.get("Name", "") or cat.get("name", "")
            if cat_id is None or not name:
                continue
            try:
                cat_id = int(cat_id)
            except (TypeError, ValueError):
                continue
            normalized = normalize_category_name(name)
            raw_parent = cat.get("FatherCategoryId") or cat.get("FatherCategoryID") or cat.get("ParentId")
            if raw_parent is None or raw_parent == "":
                parent_id = 0
            else:
                try:
                    parent_id = int(raw_parent)
                except (TypeError, ValueError):
                    parent_id = 0
            if parent_id == 0:
                new_departments[normalized] = {
                    "id": cat_id,
                    "name": normalized,
                    "created": False,
                }
            cat_key = f"{parent_id}::{normalized}"
            new_categories[cat_key] = {
                "id": cat_id,
                "name": normalized,
                "parent_id": parent_id,
                "level": cat.get("Level", 2),
                "created": False,
                "path": normalized,
            }
        if new_categories:
            self.categories.update(new_categories)
            self.logger.info(f"Synced {len(new_categories)} categories from VTEX")
        if new_departments:
            self.departments.update(new_departments)
            self.logger.info(f"Synced {len(new_departments)} departments from VTEX")
    
    def _evaluate_existing_brands(self) -> Dict[str, Dict[str, Any]]:
        """Evaluate existing brands in VTEX."""
        self.logger.info("Evaluating existing VTEX brands")
        existing = {}
        
        try:
            brands = self.vtex_client.list_brands()
            for brand in brands:
                if isinstance(brand, dict):
                    name = brand.get("Name", "")
                    if name:
                        normalized = normalize_brand_name(name)
                        existing[normalized] = brand
        except Exception as e:
            self.logger.warning(f"Error evaluating existing brands: {e}")
        
        self.logger.info(f"Found {len(existing)} existing brands")
        return existing
    
    def _ensure_category_active_and_visible(self, category_id: int) -> None:
        """Ensure category is active and visible in VTEX (IsActive, ShowInStoreFront, ActiveStoreFrontLink)."""
        if not category_id:
            return
        try:
            self.vtex_client.update_category(
                category_id,
                is_active=True,
                show_in_store_front=True,
                active_store_front_link=True,
                global_category_id=1,
            )
            self.logger.debug(f"Ensured category {category_id} is active and visible")
        except Exception as e:
            self.logger.warning(f"Could not set category {category_id} active/visible: {e}")
    
    def _process_product_categories(
        self,
        product: Dict[str, Any],
        existing_categories: Dict[str, Dict[str, Any]]
    ):
        """Process categories for a single product."""
        categories_list = product.get("categories", [])
        
        if not categories_list:
            # Try old format
            category = product.get("category", {})
            if category:
                categories_list = [category]
            else:
                return
        
        if not categories_list:
            return
        
        # First category becomes department
        dept_info = categories_list[0]
        dept_name = normalize_category_name(dept_info.get("Name", "Default"))
        
        # Create/get department
        if dept_name not in self.departments:
            # Check if exists in VTEX
            if dept_name in existing_categories:
                dept_data = existing_categories[dept_name]
                dept_id = dept_data.get("Id")
                if dept_id:
                    self.departments[dept_name] = {
                        "id": dept_id,
                        "name": dept_name,
                        "created": False
                    }
                    self.logger.debug(f"Using existing department: {dept_name} (ID: {dept_id})")
            else:
                # Create new department
                try:
                    print(f"     📁 Creating department: {dept_name}")
                    dept = self.vtex_client.create_department(dept_name)
                    dept_id = dept.get("Id") if isinstance(dept, dict) else None
                    if dept_id:
                        self.departments[dept_name] = {
                            "id": dept_id,
                            "name": dept_name,
                            "created": True
                        }
                        self.logger.info(f"Created department: {dept_name} (ID: {dept_id})")
                    else:
                        self.logger.warning(f"Could not get department ID for: {dept_name}")
                except Exception as e:
                    self.logger.error(f"Error creating department {dept_name}: {e}")
                    # Creation can fail for transient or duplicate reasons.
                    # Re-check VTEX and continue if the department actually exists.
                    try:
                        refreshed_existing = self._evaluate_existing_categories()
                        dept_data = refreshed_existing.get(dept_name)
                        dept_id = dept_data.get("Id") if isinstance(dept_data, dict) else None
                        if dept_id:
                            self.departments[dept_name] = {
                                "id": dept_id,
                                "name": dept_name,
                                "created": False
                            }
                            self.logger.info(
                                f"Recovered existing department after create error: {dept_name} (ID: {dept_id})"
                            )
                    except Exception as refresh_error:
                        self.logger.warning(
                            f"Could not refresh department lookup after create error for {dept_name}: {refresh_error}"
                        )
        
        # Create category tree
        dept_entry = self.departments.get(dept_name)
        parent_id = dept_entry.get("id") if isinstance(dept_entry, dict) else None
        if not parent_id:
            self.logger.error(
                f"Skipping product categories because department '{dept_name}' is unavailable"
            )
            return
        self._ensure_category_active_and_visible(parent_id)
        
        # If only one category, department serves as category
        if len(categories_list) == 1:
            cat_key = f"{parent_id}::{dept_name}"
            if cat_key not in self.categories:
                self.categories[cat_key] = {
                    "id": parent_id,
                    "name": dept_name,
                    "parent_id": None,
                    "level": 1,
                    "created": False,
                    "path": dept_name
                }
        else:
            # Process remaining categories (skip first, it's the department)
            for cat_info in categories_list[1:]:
                cat_name = normalize_category_name(cat_info.get("Name", ""))
                if not cat_name:
                    continue
                
                cat_key = f"{parent_id}::{cat_name}"
                
                if cat_key not in self.categories:
                    # Check if exists in VTEX
                    existing_cat = None
                    for existing_name, existing_data in existing_categories.items():
                        if existing_name == cat_name:
                            existing_cat = existing_data
                            break
                    
                    if existing_cat:
                        cat_id = existing_cat.get("Id")
                        if cat_id:
                            self._ensure_category_active_and_visible(cat_id)
                            self.categories[cat_key] = {
                                "id": cat_id,
                                "name": cat_name,
                                "parent_id": parent_id,
                                "level": cat_info.get("Level", 2),
                                "created": False,
                                "path": f"{dept_name} > {cat_name}"
                            }
                            parent_id = cat_id
                            continue
                    
                    # Create new category
                    try:
                        print(f"     📂 Creating category: {cat_name} (Level {cat_info.get('Level', 2)})")
                        cat = self.vtex_client.create_category(
                            cat_name,
                            father_category_id=parent_id
                        )
                        cat_id = cat.get("Id") if isinstance(cat, dict) else None
                        if cat_id:
                            self._ensure_category_active_and_visible(cat_id)
                            self.categories[cat_key] = {
                                "id": cat_id,
                                "name": cat_name,
                                "parent_id": parent_id,
                                "level": cat_info.get("Level", 2),
                                "created": True,
                                "path": f"{dept_name} > {cat_name}"
                            }
                            parent_id = cat_id
                            self.logger.info(f"Created category: {cat_name} (ID: {cat_id})")
                        else:
                            self.logger.warning(f"Could not get category ID for: {cat_name}")
                    except Exception as e:
                        self.logger.error(f"Error creating category {cat_name}: {e}")
                else:
                    parent_id = self.categories[cat_key]["id"]
                    self._ensure_category_active_and_visible(parent_id)
        
        time.sleep(0.2)  # Rate limiting
    
    def _process_product_brand(
        self,
        product: Dict[str, Any],
        existing_brands: Dict[str, Dict[str, Any]]
    ):
        """Process brand for a single product."""
        brand = product.get("brand", {})
        brand_name = normalize_brand_name(brand.get("Name", "Default"))
        
        if not brand_name or brand_name == "Default":
            return
        
        if brand_name not in self.brands:
            # Check if exists in VTEX
            if brand_name in existing_brands:
                brand_data = existing_brands[brand_name]
                brand_id = brand_data.get("Id")
                if brand_id:
                    self.brands[brand_name] = {
                        "id": brand_id,
                        "name": brand_name,
                        "created": False
                    }
                    self.logger.debug(f"Using existing brand: {brand_name} (ID: {brand_id})")
            else:
                # Create new brand
                try:
                    print(f"     🏷️  Creating brand: {brand_name}")
                    brand_obj = self.vtex_client.create_brand(brand_name)
                    brand_id = brand_obj.get("Id") if isinstance(brand_obj, dict) else None
                    if brand_id:
                        self.brands[brand_name] = {
                            "id": brand_id,
                            "name": brand_name,
                            "created": True
                        }
                        self.logger.info(f"Created brand: {brand_name} (ID: {brand_id})")
                    else:
                        self.logger.warning(f"Could not get brand ID for: {brand_name}")
                except Exception as e:
                    self.logger.error(f"Error creating brand {brand_name}: {e}")
        
        time.sleep(0.2)  # Rate limiting
    
    def _format_output(self) -> Dict[str, Any]:
        """Format output JSON."""
        return {
            "departments": self.departments,
            "categories": self.categories,
            "brands": self.brands,
            "summary": {
                "total_departments": len(self.departments),
                "total_categories": len(self.categories),
                "total_brands": len(self.brands),
                "departments_created": sum(1 for d in self.departments.values() if d.get("created")),
                "categories_created": sum(1 for c in self.categories.values() if c.get("created")),
                "brands_created": sum(1 for b in self.brands.values() if b.get("created"))
            }
        }
    
    def get_category_id_for_product(self, product: Dict[str, Any]) -> Optional[int]:
        """
        Get the category ID for a product based on its category hierarchy.
        Tries direct department match first, then fallback: try each department as root
        and match the full path (so product [Linhas, BARES, X] resolves under department Início).
        """
        categories_list = product.get("categories", [])
        if not categories_list:
            category = product.get("category", {})
            if category:
                categories_list = [category]
        if not categories_list:
            return None

        skip_names = {"home", "root", "default"}

        # 1) Direct: first category is a department
        dept_name = normalize_category_name(categories_list[0].get("Name", "Default"))
        if dept_name not in skip_names and dept_name in self.departments:
            parent_id = self.departments[dept_name]["id"]
            if len(categories_list) == 1:
                return parent_id
            for cat_info in categories_list[1:]:
                cat_name = normalize_category_name(cat_info.get("Name", ""))
                if not cat_name or cat_name in skip_names:
                    continue
                cat_key = f"{parent_id}::{cat_name}"
                if cat_key in self.categories:
                    parent_id = self.categories[cat_key]["id"]
                else:
                    break
            return parent_id

        # 2) Fallback: try each department as root and match full path (case-insensitive)
        for _dept_name, dept_data in self.departments.items():
            parent_id = dept_data["id"]
            matched_any = False
            for cat_info in categories_list:
                cat_name = (cat_info.get("Name") or "").strip()
                if not cat_name or cat_name.lower() in skip_names:
                    continue
                cat_name_norm = normalize_category_name(cat_name)
                cat_key = f"{parent_id}::{cat_name_norm}"
                if cat_key in self.categories:
                    parent_id = self.categories[cat_key]["id"]
                    matched_any = True
                    continue
                # Department name might match this level (single-level dept)
                if (dept_data.get("name") or "").strip().lower() == cat_name_norm.lower():
                    parent_id = dept_data["id"]
                    matched_any = True
                else:
                    break
            if matched_any:
                return parent_id

        return None

    def _longest_path_prefix(self, categories_list: List[Dict[str, Any]]) -> tuple:
        """
        Find the longest path prefix that exists in the tree. Returns (parent_id, last_matched_index+1)
        so that categories_list[returned_index:] are the part we need to create. Returns (None, 0) if
        no department matches.
        """
        skip_names = {"home", "root", "default"}
        best_parent = None
        best_index = -1
        for _dept_name, dept_data in self.departments.items():
            parent_id = dept_data["id"]
            idx = 0
            for cat_info in categories_list:
                cat_name = (cat_info.get("Name") or "").strip()
                if not cat_name or cat_name.lower() in skip_names:
                    continue
                cat_name_norm = normalize_category_name(cat_name)
                cat_key = f"{parent_id}::{cat_name_norm}"
                if cat_key in self.categories:
                    parent_id = self.categories[cat_key]["id"]
                    idx += 1
                    continue
                if (dept_data.get("name") or "").strip().lower() == cat_name_norm.lower():
                    parent_id = dept_data["id"]
                    idx += 1
                else:
                    break
            if idx > best_index:
                best_index = idx
                best_parent = parent_id
        return (best_parent, best_index)

    def _create_category_chain(
        self,
        categories_list: List[Dict[str, Any]],
        start_index: int,
        parent_id: int,
        existing_by_parent: Optional[Dict[tuple, Dict[str, Any]]] = None,
        path_prefix: str = ""
    ) -> Optional[int]:
        """Create categories from categories_list[start_index:] under parent_id. Returns leaf category id.
        existing_by_parent: (parent_id, normalized_name) -> cat dict, so we only reuse when under correct parent.
        """
        existing_by_parent = existing_by_parent or {}
        for i in range(start_index, len(categories_list)):
            cat_info = categories_list[i]
            cat_name = normalize_category_name(cat_info.get("Name", ""))
            if not cat_name:
                continue
            cat_key = f"{parent_id}::{cat_name}"
            if cat_key in self.categories:
                parent_id = self.categories[cat_key]["id"]
                self._ensure_category_active_and_visible(parent_id)
                continue
            parent_id_int = int(parent_id) if parent_id is not None else 0
            existing_cat = existing_by_parent.get((parent_id_int, cat_name))
            if existing_cat:
                cat_id = existing_cat.get("Id")
                if cat_id:
                    self._ensure_category_active_and_visible(cat_id)
                    self.categories[cat_key] = {
                        "id": cat_id,
                        "name": cat_name,
                        "parent_id": parent_id,
                        "level": cat_info.get("Level", 2),
                        "created": False,
                        "path": f"{path_prefix} > {cat_name}".strip(" >") or cat_name,
                    }
                    parent_id = cat_id
                    continue
            try:
                print(f"     📂 Creating category: {cat_name} (Level {cat_info.get('Level', 2)})")
                cat = self.vtex_client.create_category(cat_name, father_category_id=parent_id)
                cat_id = cat.get("Id") if isinstance(cat, dict) else None
                if cat_id:
                    self._ensure_category_active_and_visible(cat_id)
                    self.categories[cat_key] = {
                        "id": cat_id,
                        "name": cat_name,
                        "parent_id": parent_id,
                        "level": cat_info.get("Level", 2),
                        "created": True,
                        "path": f"{path_prefix} > {cat_name}".strip(" >") or cat_name,
                    }
                    parent_id = cat_id
                    self.logger.info(f"Created category: {cat_name} (ID: {cat_id})")
                else:
                    self.logger.warning(f"Could not get category ID for: {cat_name}")
                    return None
            except Exception as e:
                self.logger.error(f"Error creating category {cat_name}: {e}")
                return None
            path_prefix = f"{path_prefix} > {cat_name}".strip(" >")
            time.sleep(0.2)
        return parent_id

    def ensure_category_for_product(self, product: Dict[str, Any]) -> Tuple[Optional[int], Dict[str, Any]]:
        """
        Ensure the product's category path exists in VTEX (create if missing), then return
        the leaf category ID and the updated category tree.
        Call this when get_category_id_for_product returns None before creating the product.
        Re-syncs tree from VTEX first so we reassess what exists and create only what is missing
        under the correct department.
        Returns:
            Tuple of (category_id or None, updated_tree dict for vtex_category_tree)
        """
        from ..utils.state_manager import save_state

        categories_list = product.get("categories", [])
        if not categories_list:
            category = product.get("category", {})
            if category:
                categories_list = [category]
        if not categories_list:
            self.logger.warning("ensure_category_for_product: product has no categories")
            return None, self._format_output()

        # Re-sync from VTEX so we have current state and create under the right department
        self._sync_tree_from_vtex()

        # Already resolvable with current tree?
        category_id = self.get_category_id_for_product(product)
        if category_id is not None:
            self._ensure_category_active_and_visible(category_id)
            return category_id, self._format_output()

        self.logger.info(
            "Category path not found in tree; creating missing categories for product."
        )
        print("       📂 Category path missing in VTEX; creating/finding categories...")
        parent_id, start_index = self._longest_path_prefix(categories_list)
        if parent_id is None or start_index < 0:
            # No path at all - create from first category as department
            existing_categories = self._evaluate_existing_categories()
            self._process_product_categories(product, existing_categories)
        else:
            # Create only the missing tail under the right parent
            existing_by_parent = self._existing_categories_by_parent()
            path_prefix = ""
            for j in range(0, start_index):
                path_prefix = f"{path_prefix} > {normalize_category_name(categories_list[j].get('Name', ''))}".strip(" >")
            leaf_id = self._create_category_chain(
                categories_list, start_index, parent_id, existing_by_parent=existing_by_parent, path_prefix=path_prefix
            )
            if leaf_id is None:
                self.logger.warning("_create_category_chain did not create; trying full path from first as department.")
                existing_categories = self._evaluate_existing_categories()
                self._process_product_categories(product, existing_categories)
        updated = self._format_output()
        save_state("vtex_category_tree", updated)
        category_id = self.get_category_id_for_product(product)
        if category_id is not None:
            self._ensure_category_active_and_visible(category_id)
            print(f"       ✓ Category resolved (ID: {category_id})")
        else:
            self.logger.warning("Category still not resolved after create attempts; product may need manual category.")
        return category_id, updated
    
    def get_brand_id(self, brand_name: str) -> Optional[int]:
        """
        Get brand ID by name.
        
        Args:
            brand_name: Brand name
            
        Returns:
            Brand ID or None
        """
        normalized = normalize_brand_name(brand_name)
        if not normalized:
            return None
        
        target = normalized.strip().lower()
        
        # Match against both stored keys and brand names in a case-insensitive way
        for brand_key, brand_data in self.brands.items():
            key_name = str(brand_key).strip().lower()
            data_name = str(brand_data.get("name", "")).strip().lower()
            
            if target == key_name or target == data_name:
                return brand_data.get("id")
        
        return None

