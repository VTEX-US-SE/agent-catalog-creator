"""Migration Agent - Coordinates the VTEX migration workflow."""
import os
import json
from typing import Dict, Any, Optional

from .legacy_site_agent import LegacySiteAgent
from .vtex_category_tree_agent import VTEXCategoryTreeAgent
from .vtex_product_sku_agent import VTEXProductSKUAgent
from .vtex_image_agent import VTEXImageAgent
from ..clients.vtex_client import VTEXClient
from ..utils.state_manager import save_state, load_state, STATE_DIR
from ..utils.logger import get_agent_logger
from ..tools.gemini_mapper import analyze_structure_from_sample


class MigrationAgent:
    """Migration coordinator agent for VTEX catalog migration."""
    
    def __init__(self):
        self.logger = get_agent_logger("migration_agent")
        
        # Initialize agents
        self.legacy_site_agent = LegacySiteAgent()
        self.vtex_client = None  # Initialize when needed
        self.vtex_category_tree_agent = None
        self.vtex_product_sku_agent = None
        self.vtex_image_agent = None
        
        # Workflow state
        self.workflow_state = {}
    
    def run_full_workflow(self):
        """Execute the complete migration workflow."""
        print("\n" + "="*60)
        print("🤖 VTEX CATALOG MIGRATION AGENT")
        print("="*60)
        self.logger.info("Starting full workflow")
        
        try:
            # Step 1: Discovery
            target_url = self.discovery_phase()
            
            # Step 2: Mapping
            product_urls = self.mapping_phase()
            
            # Step 3: Extraction (sample first)
            print("\n💡 Extracting 1 sample product for mapping validation...")
            legacy_site_data = self.extraction_phase(sample_size=1)
            
            # Step 4: Sampling
            selected_urls = self.sampling_phase()
            
            # Re-extract selected products if needed
            if len(selected_urls) > 1:
                print(f"\n📥 Extracting {len(selected_urls)} selected products...")
                self.legacy_site_agent.product_urls = selected_urls
                legacy_site_data = self.legacy_site_agent.extract_all_products()
            
            # Step 5: Reporting
            self.reporting_phase(legacy_site_data)
            
            # Step 6: Execution (requires approval)
            self.execution_phase(legacy_site_data, require_approval=True)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Workflow interrupted by user.")
            self.logger.warning("Workflow interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Error in workflow: {e}")
            self.logger.error(f"Error in workflow: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
    
    def discovery_phase(self) -> str:
        """Step 1: Discovery - Get target website URL."""
        print("\n" + "="*60)
        print("📋 STEP 1: DISCOVERY")
        print("="*60)
        
        state = load_state("discovery")
        if state and state.get("target_url"):
            print(f"✅ Found existing discovery state.")
            target_url = state["target_url"]
            print(f"   Target URL: {target_url}")
            use_existing = input("\n   Use existing target URL? (y/n): ").strip().lower()
            if use_existing == "y":
                return self.legacy_site_agent.discover_target_url(target_url)
        
        return self.legacy_site_agent.discover_target_url()
    
    def mapping_phase(self) -> list:
        """Step 2: Mapping - Find product URLs."""
        print("\n" + "="*60)
        print("🗺️  STEP 2: MAPPING - Finding Product URLs")
        print("="*60)
        
        state = load_state("mapping")
        if state and state.get("product_urls"):
            print(f"✅ Found existing mapping state with {len(state['product_urls'])} URLs.")
            use_existing = input("\n   Use existing product URLs? (y/n): ").strip().lower()
            if use_existing == "y":
                self.legacy_site_agent.product_urls = state["product_urls"]
                print(f"   ✅ Loaded {len(self.legacy_site_agent.product_urls)} product URLs")
                return self.legacy_site_agent.product_urls
        
        return self.legacy_site_agent.map_product_urls()
    
    def extraction_phase(self, sample_size: int = 1) -> Dict[str, Any]:
        """Step 3: Extraction - Extract and map products."""
        print("\n" + "="*60)
        print("🔬 STEP 3: DATA EXTRACTION & ALIGNMENT")
        print("="*60)
        
        return self.legacy_site_agent.extract_products(
            sample_size=sample_size,
            enable_iterative_refinement=True
        )
    
    def sampling_phase(self) -> list:
        """Step 4: Sampling - Select products to import."""
        print("\n" + "="*60)
        print("📊 STEP 4: SAMPLING & STRATEGY")
        print("="*60)
        
        print(f"\n📈 Total product URLs available: {len(self.legacy_site_agent.product_urls)}")
        
        import_count = input("\nHow many products would you like to import for this phase? (or 'all' for all): ").strip()
        
        if import_count.lower() == "all":
            selected_urls = self.legacy_site_agent.product_urls
        else:
            try:
                count = int(import_count)
                if count >= len(self.legacy_site_agent.product_urls):
                    selected_urls = self.legacy_site_agent.product_urls
                else:
                    step = len(self.legacy_site_agent.product_urls) // count
                    selected_urls = self.legacy_site_agent.product_urls[::step][:count]
            except ValueError:
                print("⚠️  Invalid input, using first 10 products")
                selected_urls = self.legacy_site_agent.product_urls[:10]
        
        save_state("sampling", {
            "selected_count": len(selected_urls),
            "selected_urls": selected_urls
        })
        
        print(f"✅ Selected {len(selected_urls)} products for import")
        return selected_urls
    
    def reporting_phase(self, legacy_site_data: Dict[str, Any]):
        """Step 5: Reporting - Generate migration plan."""
        print("\n" + "="*60)
        print("📄 STEP 5: REPORTING")
        print("="*60)
        
        if not legacy_site_data or not legacy_site_data.get("products"):
            print("⚠️  No extracted products found.")
            return
        
        # Analyze structure
        print("\n📊 Analyzing catalog structure...")
        sample_data = [p.get("mapped_data", p) for p in legacy_site_data.get("products", [])[:5]]
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        structure = analyze_structure_from_sample(sample_data, gemini_api_key)
        
        # Generate report
        report_lines = [
            "# VTEX Catalog Migration Plan",
            "",
            f"**Target Website:** {legacy_site_data.get('target_url', 'N/A')}",
            f"**Total Product URLs Found:** {legacy_site_data.get('metadata', {}).get('total_urls_found', 0)}",
            f"**Products Extracted:** {len(legacy_site_data.get('products', []))}",
            "",
            "## Catalog Structure",
            "",
            "### Departments",
        ]
        
        departments = structure.get("departments", [])
        for dept in departments:
            report_lines.append(f"- {dept}")
        
        report_lines.extend(["", "### Categories"])
        categories = structure.get("categories", [])
        for cat in categories:
            cat_name = cat.get("Name", "") if isinstance(cat, dict) else str(cat)
            dept = cat.get("Department", "") if isinstance(cat, dict) else ""
            report_lines.append(f"- {cat_name}" + (f" (Department: {dept})" if dept else ""))
        
        report_lines.extend(["", "### Brands"])
        brands = structure.get("brands", [])
        for brand in brands:
            report_lines.append(f"- {brand}")
        
        report_lines.extend(["", "### Specification Groups"])
        spec_groups = structure.get("specification_groups", [])
        for spec in spec_groups:
            report_lines.append(f"- {spec}")
        
        report_lines.extend([
            "",
            "## Product Counts",
            "",
            f"- **Total Products:** {structure.get('total_products', 0)}",
            f"- **Has Variations:** {structure.get('product_patterns', {}).get('has_variations', False)}",
            "",
            "## Next Steps",
            "",
            "1. Review the catalog structure above",
            "2. Confirm Specification Groups match your VTEX setup",
            "3. Type 'APPROVED' to begin execution",
            ""
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save report in state folder
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        report_path = STATE_DIR / "final_plan.md"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        print(f"\n✅ Report generated: {report_path}")
        print("\n" + "="*60)
        print("REPORT PREVIEW")
        print("="*60)
        print(report_content)
        
        save_state("reporting", {
            "structure": structure,
            "report_path": str(report_path)
        })
    
    def execution_phase(self, legacy_site_data: Dict[str, Any], require_approval: bool = True):
        """Step 6: Execution - Create catalog in VTEX."""
        if require_approval:
            while True:
                print("\n" + "="*60)
                print("⚠️  READY TO EXECUTE")
                print("="*60)
                print("\nOptions:")
                print("  - Type 'APPROVED' to proceed with execution")
                print("  - Type 'RETRY' to regenerate the report and review again")
                print("  - Type 'CANCEL' to end execution")
                
                approval = input("\nWhat would you like to do? ").strip().upper()
                
                if approval == "APPROVED":
                    break
                elif approval == "CANCEL":
                    print("\n❌ Execution cancelled by user.")
                    return
                elif approval == "RETRY":
                    print("\n🔄 Regenerating report...")
                    self.reporting_phase(legacy_site_data)
                    # Loop back to ask for approval again
                    continue
                else:
                    print(f"\n⚠️  Invalid option: '{approval}'. Please type 'APPROVED', 'RETRY', or 'CANCEL'.")
                    continue
        
        print("\n" + "="*60)
        print("🚀 STEP 6: EXECUTION - VTEX Catalog Import")
        print("="*60)
        
        # Initialize VTEX client
        try:
            self.vtex_client = VTEXClient()
            print("✅ VTEX client initialized")
            self.logger.info("VTEX client initialized")
        except Exception as e:
            print(f"❌ VTEX credentials not configured: {e}")
            print("   Set VTEX_ACCOUNT_NAME, VTEX_APP_KEY, and VTEX_APP_TOKEN in .env")
            self.logger.error(f"VTEX credentials not configured: {e}")
            return
        
        # Initialize VTEX agents (product agent can call category tree agent to create missing categories)
        self.vtex_category_tree_agent = VTEXCategoryTreeAgent(self.vtex_client)
        # Specifications are disabled - no specification fields will be created or set
        self.vtex_product_sku_agent = VTEXProductSKUAgent(
            self.vtex_client,
            category_tree_agent=self.vtex_category_tree_agent
        )
        self.vtex_image_agent = VTEXImageAgent(self.vtex_client)
        
        # Step 1: Create category tree
        print("\n📂 Creating category tree...")
        vtex_category_tree = self.vtex_category_tree_agent.create_category_tree(legacy_site_data)
        
        # Step 2: Create products, SKUs, and associate images (new ordering)
        # Note: Specifications are disabled - no specification fields will be created or set in VTEX
        print("\n📦 Creating products, SKUs, and associating images...")
        print("   Order: Product → SKU → Images")
        
        products = legacy_site_data.get("products", [])
        print(f"\n📦 Processing {len(products)} products...")
        
        all_image_results = {}
        
        for i, product_data in enumerate(products, 1):
            print(f"\n   [{i}/{len(products)}] Processing product...")
            self.logger.info(f"Processing product {i}/{len(products)}")
            
            try:
                # Create product (specifications disabled - pass empty dict)
                product_info = self.vtex_product_sku_agent.create_single_product(
                    product_data,
                    vtex_category_tree,
                    {"specification_fields": {}, "summary": {"fields_created": 0}}
                )
                
                # If category tree was updated (e.g. missing categories created), use it for next products
                if product_info and product_info.get("vtex_category_tree") is not None:
                    vtex_category_tree = product_info["vtex_category_tree"]
                
                if not product_info:
                    self.logger.warning(f"Failed to create product {i}, skipping")
                    continue
                
                product_id = product_info["id"]
                product_url = product_data.get("url", f"product_{product_id}")
                
                # Get SKUs for this product
                skus = product_data.get("skus", [])
                if not skus:
                    # Create default SKU
                    skus = [{
                        "Name": "Default",
                        "EAN": f"EAN{product_id}",
                        "IsActive": True
                    }]
                
                # Get images for this product
                images = product_data.get("images", [])
                
                # Create each SKU and immediately associate images
                for sku_data in skus:
                    # Create SKU
                    sku_info = self.vtex_product_sku_agent.create_single_sku(
                        product_id=product_id,
                        product_url=product_url,
                        sku_data=sku_data
                    )
                    
                    if not sku_info:
                        self.logger.warning(f"Failed to create SKU for product {product_id}, skipping")
                        continue
                    
                    sku_id = sku_info["id"]
                    sku_name = sku_info["name"]
                    
                    # Step 1: Associate images with this SKU (VTEX requires files before SKU can be active)
                    had_images = False
                    if images:
                        image_result = self.vtex_image_agent.associate_images_with_sku(
                            sku_id=sku_id,
                            sku_name=sku_name,
                            image_urls=images
                        )
                        all_image_results[str(sku_id)] = image_result
                        had_images = (image_result.get("total_associated") or 0) > 0
                    else:
                        self.logger.info(f"No images found for SKU {sku_id}")
                    
                    # Step 2: Activate SKU only after images are associated (VTEX requires files before IsActive=true)
                    if had_images:
                        try:
                            self.vtex_client.update_sku(sku_id, is_active=True)
                            print(f"       ✓ SKU activated (IsActive=true)")
                        except Exception as activate_error:
                            self.logger.warning(f"Could not activate SKU {sku_id}: {activate_error}")
                            print(f"       ⚠️  Failed to activate SKU: {activate_error}")
                    else:
                        print(f"       ℹ️  SKU left inactive (no images; VTEX requires files before activating)")
                    
                    # Step 3: Set price for this SKU
                    # Order: Create SKU > Add images > Add price > Add inventory
                    # Price from website is set as basePrice with markup=0
                    try:
                        price_value = sku_data.get("Price") or 0
                        list_price_value = sku_data.get("ListPrice") or price_value
                        self.vtex_client.set_sku_price(sku_id, price_value, list_price_value)
                        print(f"       💰 Price set: {price_value} (basePrice, markup=0)")
                    except Exception as price_error:
                        self.logger.warning(f"Could not set price for SKU {sku_id}: {price_error}")
                        print(f"       ⚠️  Failed to set price: {price_error}")
                    
                    # Step 4: Set inventory for this SKU in all warehouses
                    # Inventory is set to 100 for all available warehouses
                    try:
                        inventory_results = self.vtex_client.set_sku_inventory_all_warehouses(
                            sku_id=sku_id,
                            quantity=100  # Set to 100 for all warehouses
                        )
                        successful_warehouses = sum(1 for r in inventory_results.values() if r.get("success", False))
                        print(f"       📦 Inventory set to 100 in {successful_warehouses}/{len(inventory_results)} warehouse(s)")
                    except Exception as inventory_error:
                        self.logger.warning(f"Could not set inventory for SKU {sku_id}: {inventory_error}")
                        print(f"       ⚠️  Failed to set inventory: {inventory_error}")
                
            except Exception as e:
                self.logger.error(f"Error processing product {i}: {e}", exc_info=True)
                print(f"     ⚠️  Error processing product: {e}")
                continue
        
        # Format outputs
        vtex_products = self.vtex_product_sku_agent._format_output()
        
        # Save product/SKU state
        save_state("vtex_products_skus", vtex_products)
        
        # Format image results
        vtex_images = self.vtex_image_agent._format_output()
        
        # Save image state
        save_state("vtex_images", vtex_images)
        
        # Save execution summary
        save_state("execution", {
            "departments_created": vtex_category_tree.get("summary", {}).get("departments_created", 0),
            "categories_created": vtex_category_tree.get("summary", {}).get("categories_created", 0),
            "brands_created": vtex_category_tree.get("summary", {}).get("brands_created", 0),
            "products_created": vtex_products.get("summary", {}).get("products_created", 0),
            "skus_created": vtex_products.get("summary", {}).get("skus_created", 0),
            "images_uploaded": vtex_images.get("summary", {}).get("total_images_associated", vtex_images.get("summary", {}).get("total_images_uploaded", 0))
        })
        
        print("\n" + "="*60)
        print("✅ EXECUTION COMPLETE")
        print("="*60)
        print(f"   Departments: {vtex_category_tree.get('summary', {}).get('total_departments', 0)}")
        print(f"   Categories: {vtex_category_tree.get('summary', {}).get('total_categories', 0)}")
        print(f"   Brands: {vtex_category_tree.get('summary', {}).get('total_brands', 0)}")
        print(f"   Products: {vtex_products.get('summary', {}).get('total_products', 0)}")
        print(f"   SKUs: {vtex_products.get('summary', {}).get('total_skus', 0)}")
        print(f"   Images: {vtex_images.get('summary', {}).get('total_images_uploaded', 0)}")
        print(f"   Note: Specifications are disabled - no specification fields created or set")
        
        self.logger.info("Execution phase complete")

