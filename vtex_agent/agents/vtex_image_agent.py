"""VTEX Image Enrichment Agent - Uses existing image URLs and associates with SKUs."""
from typing import Dict, Any, List, Optional
import time
import os

from ..clients.vtex_client import VTEXClient
from ..utils.state_manager import save_state, load_state
from ..utils.logger import get_agent_logger
from ..tools.image_manager import process_and_upload_images_to_github


class VTEXImageAgent:
    """
    Agent responsible for the final enrichment of VTEX SKUs.
    
    This agent:
    1. Processes product images from the legacy site
    2. Downloads and renames them using format: [SkuId]_[SequenceNumber]
    3. Uses existing image URLs (no GitHub upload)
    4. Associates images with SKUs in VTEX using the VTEX API
    """
    
    def __init__(self, vtex_client: Optional[VTEXClient] = None):
        self.logger = get_agent_logger("vtex_image_agent")
        self.vtex_client = vtex_client or VTEXClient()
        
        # Track image associations per SKU
        self.sku_image_associations = {}
    
    def enrich_skus_with_images(
        self,
        legacy_site_data: Dict[str, Any],
        vtex_products_skus: Dict[str, Any],
        github_repo_path: str = "images"
    ) -> Dict[str, Any]:
        """
        Enrich VTEX SKUs with images from the legacy site.
        
        This method:
        1. Reads VtexSkuId from previous agent's output
        2. Downloads images from legacy site
        3. Renames images: [SkuId]_[SequenceNumber]
        4. Uses existing image URLs directly
        5. Associates images with SKU via VTEX API
        
        Args:
            legacy_site_data: Output from Legacy Site Agent (contains image URLs)
            vtex_products_skus: Output from VTEX Product/SKU Agent (contains SKU IDs)
            github_repo_path: Path within GitHub repository for images
            
        Returns:
            Dictionary with image association results per SKU
        """
        self.logger.info("Starting SKU image enrichment")
        
        # Try to load from state (but allow re-processing if needed)
        state = load_state("vtex_images")
        if state and state.get("sku_image_associations"):
            print("   ℹ️  Found existing image associations in state")
            # Show summary of what's in state
            state_summary = state.get("summary", {})
            print(f"      Total SKUs: {state_summary.get('total_skus', 0)}")
            print(f"      Images associated: {state_summary.get('total_images_associated', 0)}")
            print(f"      Images failed: {state_summary.get('total_images_failed', 0)}")
            
            # Check if we should re-process (if any SKU has failed status)
            failed_skus = [
                sku_id for sku_id, assoc in state.get("sku_image_associations", {}).items()
                if assoc.get("status") == "failed"
            ]
            if failed_skus:
                print(f"   🔄 Found {len(failed_skus)} failed SKU(s), re-processing...")
                print(f"      Failed SKUs: {', '.join(failed_skus)}")
                # Clear failed associations to re-process them
                self.sku_image_associations = {}
            else:
                print("   ✅ All SKUs processed successfully, using cached results")
                print("   ℹ️  To force re-processing, delete state/vtex_images.json")
                self.logger.info("Loaded image associations from state")
                self.sku_image_associations = state.get("sku_image_associations", {})
                return self._format_output()
        
        # Get products from legacy site
        products = legacy_site_data.get("products", [])
        vtex_products_dict = vtex_products_skus.get("products", {})
        
        # Create mapping from product URL to VTEX product data
        url_to_vtex_product = {}
        for product_url, product_data in vtex_products_dict.items():
            url_to_vtex_product[product_url] = product_data
        
        total_skus_processed = 0
        total_images_associated = 0
        total_images_failed = 0
        
        # Process each product
        for product_data in products:
            product_url = product_data.get("url", "")
            vtex_product = url_to_vtex_product.get(product_url)
            
            if not vtex_product:
                self.logger.warning(f"Could not find VTEX product for URL: {product_url}")
                continue
            
            # Get SKUs for this product
            skus = vtex_product.get("skus", [])
            if not skus:
                self.logger.warning(f"No SKUs found for product URL: {product_url}")
                continue
            
            # Get images from legacy site
            images = product_data.get("images", [])
            if not images:
                self.logger.info(f"No images found for product URL: {product_url}")
                continue
            
            # Process each SKU
            for sku_data in skus:
                sku_id = sku_data.get("id")
                if not sku_id:
                    self.logger.warning(f"SKU data missing ID: {sku_data}")
                    continue
                
                sku_name = sku_data.get("name", "Product Image")
                print(f"\n   🖼️  Processing images for SKU ID {sku_id} ({sku_name})...")
                self.logger.info(f"Processing {len(images)} images for SKU {sku_id}")
                
                total_skus_processed += 1
                
                # Prepare image URLs (no GitHub upload)
                uploaded_images = process_and_upload_images_to_github(
                    image_urls=images,
                    sku_id=sku_id,
                    repo_path=github_repo_path
                )
                
                # Associate images with SKU in VTEX
                associated_images = []
                failed_count = 0
                
                for idx, img_info in enumerate(uploaded_images, start=1):
                    if not img_info.get("url"):
                        failed_count += 1
                        total_images_failed += 1
                        continue
                    
                    raw_github_url = img_info["url"]
                    file_name = img_info["name"]
                    is_main = (idx == 1)  # First image is main
                    
                    try:
                        print(f"     [{idx}/{len(uploaded_images)}] Associating image with VTEX SKU...")
                        
                        result = self.vtex_client.associate_sku_image(
                            sku_id=sku_id,
                            image_url=raw_github_url,
                            file_name=file_name,
                            is_main=is_main,
                            label=sku_name
                        )
                        
                        if result:
                            associated_images.append({
                                "url": raw_github_url,
                                "name": file_name,
                                "sequence": img_info["sequence"],
                                "is_main": is_main,
                                "status": "associated",
                                "vtex_response": result
                            })
                            total_images_associated += 1
                            self.logger.debug(
                                f"Successfully associated image {file_name} with SKU {sku_id}"
                            )
                            print(f"       ✅ Associated image {file_name} with SKU {sku_id}")
                        else:
                            associated_images.append({
                                "url": raw_github_url,
                                "name": file_name,
                                "sequence": img_info["sequence"],
                                "is_main": is_main,
                                "status": "failed",
                                "error": "Empty response from VTEX API"
                            })
                            failed_count += 1
                            total_images_failed += 1
                            print(f"       ❌ Failed to associate image {file_name}")
                        
                        time.sleep(0.3)  # Rate limiting
                        
                    except Exception as e:
                        self.logger.error(
                            f"Error associating image {file_name} with SKU {sku_id}: {e}",
                            exc_info=True
                        )
                        associated_images.append({
                            "url": raw_github_url,
                            "name": file_name,
                            "sequence": img_info["sequence"],
                            "is_main": is_main,
                            "status": "failed",
                            "error": str(e)[:200]
                        })
                        failed_count += 1
                        total_images_failed += 1
                        print(f"       ❌ Error: {str(e)[:100]}")
                
                # Store results for this SKU
                self.sku_image_associations[str(sku_id)] = {
                    "sku_id": sku_id,
                    "sku_name": sku_name,
                    "images": associated_images,
                    "total_uploaded": len([img for img in uploaded_images if img.get("status") == "uploaded"]),
                    "total_associated": len([img for img in associated_images if img.get("status") == "associated"]),
                    "total_failed": failed_count,
                    "status": "completed" if len(associated_images) > 0 else "failed"
                }
        
        # Save output
        output = self._format_output()
        output["summary"]["total_skus_processed"] = total_skus_processed
        output["summary"]["total_images_associated"] = total_images_associated
        output["summary"]["total_images_failed"] = total_images_failed
        
        save_state("vtex_images", output)
        
        self.logger.info(
            f"SKU image enrichment complete. "
            f"Processed {total_skus_processed} SKUs, "
            f"associated {total_images_associated} images, "
            f"{total_images_failed} failed"
        )
        
        return output
    
    def _format_output(self) -> Dict[str, Any]:
        """Format output JSON."""
        total_skus = len(self.sku_image_associations)
        total_images = sum(
            len(assoc.get("images", []))
            for assoc in self.sku_image_associations.values()
        )
        total_associated = sum(
            assoc.get("total_associated", 0)
            for assoc in self.sku_image_associations.values()
        )
        total_failed = sum(
            assoc.get("total_failed", 0)
            for assoc in self.sku_image_associations.values()
        )
        
        return {
            "sku_image_associations": self.sku_image_associations,
            "summary": {
                "total_skus": total_skus,
                "total_images": total_images,
                "total_images_associated": total_associated,
                "total_images_uploaded": total_associated,  # Backward compatibility
                "total_images_failed": total_failed
            }
        }
    
    def associate_images_with_sku(
        self,
        sku_id: int,
        sku_name: str,
        image_urls: List[str],
        github_repo_path: str = "images"
    ) -> Dict[str, Any]:
        """
        Associate images with a single SKU.
        
        Args:
            sku_id: VTEX SKU ID
            sku_name: SKU name for labeling
            image_urls: List of image URLs from legacy site
            github_repo_path: Path within GitHub repository for images
            
        Returns:
            Dictionary with image association results for this SKU
        """
        if not image_urls:
            self.logger.info(f"No images found for SKU {sku_id}")
            return {
                "sku_id": sku_id,
                "sku_name": sku_name,
                "images": [],
                "total_uploaded": 0,
                "total_associated": 0,
                "total_failed": 0,
                "status": "no_images"
            }
        
        print(f"     🖼️  Processing {len(image_urls)} images for SKU {sku_id}...")
        self.logger.info(f"Processing {len(image_urls)} images for SKU {sku_id}")
        
        # Prepare image URLs (no GitHub upload)
        uploaded_images = process_and_upload_images_to_github(
            image_urls=image_urls,
            sku_id=sku_id,
            repo_path=github_repo_path
        )
        
        # Associate images with SKU in VTEX
        associated_images = []
        failed_count = 0
        
        for idx, img_info in enumerate(uploaded_images, start=1):
            if not img_info.get("url"):
                # Image URL preparation failed - record the failure
                failed_count += 1
                associated_images.append({
                    "url": None,
                    "name": img_info.get("name", "unknown"),
                    "sequence": img_info.get("sequence", idx),
                    "is_main": (idx == 1),
                    "status": "failed",
                    "error": img_info.get("error", "Failed to prepare image URL"),
                    "original_url": img_info.get("original_url", "unknown")
                })
                print(f"       ❌ Skipping image {img_info.get('name', 'unknown')}: {img_info.get('error', 'Image URL preparation failed')}")
                continue
            
            raw_github_url = img_info["url"]
            file_name = img_info["name"]
            is_main = (idx == 1)  # First image is main
            
            try:
                print(f"       [{idx}/{len(uploaded_images)}] Step 3: Associating image with VTEX SKU...")
                print(f"          SKU ID: {sku_id}")
                print(f"          Image URL: {raw_github_url[:80]}...")
                print(f"          File name: {file_name}")
                print(f"          Is main: {is_main}")
                
                result = self.vtex_client.associate_sku_image(
                    sku_id=sku_id,
                    image_url=raw_github_url,
                    file_name=file_name,
                    is_main=is_main,
                    label=sku_name
                )
                
                if result:
                    associated_images.append({
                        "url": raw_github_url,
                        "name": file_name,
                        "sequence": img_info["sequence"],
                        "is_main": is_main,
                        "status": "associated",
                        "vtex_response": result
                    })
                    self.logger.debug(
                        f"Successfully associated image {file_name} with SKU {sku_id}"
                    )
                    print(f"       ✅ Association successful!")
                    print(f"          Response: {str(result)[:100]}...")
                else:
                    associated_images.append({
                        "url": raw_github_url,
                        "name": file_name,
                        "sequence": img_info["sequence"],
                        "is_main": is_main,
                        "status": "failed",
                        "error": "Empty response from VTEX API"
                    })
                    failed_count += 1
                    print(f"       ❌ Association failed: Empty response from VTEX API")
                
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                self.logger.error(
                    f"Error associating image {file_name} with SKU {sku_id}: {e}",
                    exc_info=True
                )
                associated_images.append({
                    "url": raw_github_url,
                    "name": file_name,
                    "sequence": img_info["sequence"],
                    "is_main": is_main,
                    "status": "failed",
                    "error": str(e)[:200]
                })
                failed_count += 1
                print(f"         ❌ Error: {str(e)[:100]}")
        
        # Store results for this SKU
        result = {
            "sku_id": sku_id,
            "sku_name": sku_name,
            "images": associated_images,
            "total_uploaded": len([img for img in uploaded_images if img.get("status") == "uploaded"]),
            "total_associated": len([img for img in associated_images if img.get("status") == "associated"]),
            "total_failed": failed_count,
            "status": "completed" if len(associated_images) > 0 else "failed"
        }
        
        self.sku_image_associations[str(sku_id)] = result
        
        return result
    
    # Backward compatibility method
    def upload_images(
        self,
        legacy_site_data: Dict[str, Any],
        vtex_products: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Backward compatibility method.
        
        This method now calls enrich_skus_with_images for SKU-based image association.
        """
        self.logger.info("Using backward compatibility method - redirecting to enrich_skus_with_images")
        return self.enrich_skus_with_images(legacy_site_data, vtex_products)
