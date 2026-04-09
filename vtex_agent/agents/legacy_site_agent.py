"""Legacy Site Agent - Extracts product data from source website."""
import os
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..tools.sitemap_crawler import extract_sitemap_urls, recursive_crawl_pdp_patterns, crawl_categories, build_session
from ..tools.gemini_mapper import extract_to_vtex_schema, initialize_gemini, _retry_with_exponential_backoff
from ..utils.state_manager import save_state, load_state, save_custom_prompt
from ..utils.prompt_manager import get_custom_prompt, edit_custom_prompt_interactive
from ..utils.logger import get_agent_logger
from ..utils.validation import validate_legacy_site_output


class LegacySiteAgent:
    """Agent responsible for extracting data from legacy websites."""
    
    def __init__(self):
        self.logger = get_agent_logger("legacy_site_agent")
        self.session = build_session()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Agent state
        self.target_url = None
        self.product_urls = []
        self.extracted_products = []
    
    def discover_target_url(self, target_url: Optional[str] = None) -> str:
        """
        Discover and set target website URL.
        
        Args:
            target_url: Optional target URL (if None, loads from state or prompts user)
            
        Returns:
            Target URL string
        """
        self.logger.info("Starting discovery phase")
        
        # Try to load from state
        state = load_state("discovery")
        if state and state.get("target_url"):
            self.target_url = state["target_url"]
            self.logger.info(f"Loaded target URL from state: {self.target_url}")
            return self.target_url
        
        # Use provided URL or prompt
        if target_url:
            self.target_url = target_url
        else:
            self.target_url = input("\n🌐 Enter the target website URL to migrate: ").strip()
        
        if not self.target_url.startswith(("http://", "https://")):
            self.target_url = f"https://{self.target_url}"
        
        save_state("discovery", {"target_url": self.target_url})
        self.logger.info(f"Target URL set: {self.target_url}")
        
        return self.target_url
    
    def map_product_urls(self, max_pages: int = 50) -> List[str]:
        """
        Map and discover product URLs from target website.
        
        Args:
            max_pages: Maximum pages to crawl if sitemap not found
            
        Returns:
            List of product URLs
        """
        self.logger.info("Starting mapping phase")
        
        if not self.target_url:
            raise ValueError("Target URL not set. Run discover_target_url() first.")
        
        # Try to load from state
        state = load_state("mapping")
        if state and state.get("product_urls"):
            self.product_urls = state["product_urls"]
            self.logger.info(f"Loaded {len(self.product_urls)} URLs from state")
            return self.product_urls
        
        # Try sitemap first
        self.logger.info(f"Extracting sitemap from: {self.target_url}")
        sitemap_urls = extract_sitemap_urls(self.target_url, self.session)
        
        if sitemap_urls:
            self.logger.info(f"Found {len(sitemap_urls)} URLs from sitemap")
            self.product_urls = sitemap_urls
        else:
            self.logger.info("No sitemap found. Using recursive crawler...")
            print("   🔍 Crawling website to collect ALL URLs (no filtering)...")
            self.product_urls = recursive_crawl_pdp_patterns(
                self.target_url,
                self.session,
                max_pages=max_pages
            )
            self.logger.info(f"Collected {len(self.product_urls)} URLs (all URLs, not filtered)")
        
        # Deduplicate
        self.product_urls = list(set(self.product_urls))
        
        # LLM-based URL review and refinement
        print(f"\n🤖 Reviewing {len(self.product_urls)} URLs with LLM...")
        self.product_urls = self._review_urls_with_llm(self.product_urls)
        
        save_state("mapping", {
            "target_url": self.target_url,
            "product_urls": self.product_urls,
            "url_count": len(self.product_urls)
        })
        
        self.logger.info(f"Mapping complete. Found {len(self.product_urls)} unique product URLs")
        return self.product_urls
    
    def extract_products(
        self,
        sample_size: int = 1,
        custom_instructions: Optional[str] = None,
        enable_iterative_refinement: bool = True
    ) -> Dict[str, Any]:
        """
        Extract products from URLs with optional iterative refinement.
        
        Args:
            sample_size: Number of products to extract initially
            custom_instructions: Optional custom extraction instructions
            enable_iterative_refinement: Whether to enable iterative refinement loop
            
        Returns:
            Dictionary with extracted products in standardized format
        """
        self.logger.info(f"Starting extraction phase (sample_size={sample_size})")
        
        if not self.product_urls:
            raise ValueError("No product URLs found. Run map_product_urls() first.")
        
        # Load custom prompt if not provided
        if custom_instructions is None:
            custom_instructions = get_custom_prompt()
            if custom_instructions:
                self.logger.info(f"Using custom extraction instructions ({len(custom_instructions)} chars)")
        
        # Step 1: Extract sample product for validation (always 1 product)
        print(f"\n📋 Step 1: Extracting 1 sample product for validation...")
        sample_urls = self.product_urls[:1]
        extracted_samples = self._extract_products_batch(sample_urls, custom_instructions)
        
        # Step 2: Iterative refinement loop (if enabled)
        if enable_iterative_refinement:
            extracted_samples = self._iterative_refinement_loop(
                extracted_samples,
                sample_urls,
                custom_instructions
            )
        
        # Step 3: Ask user how many products to extract
        print("\n" + "="*60)
        print("📊 PRODUCT EXTRACTION - QUANTITY SELECTION")
        print("="*60)
        print(f"\n📈 Total product URLs available: {len(self.product_urls)}")
        print(f"✅ Sample product extraction approved.")
        
        import_count = input("\nHow many products would you like to extract? (or 'all' for all): ").strip()
        
        if import_count.lower() == "all":
            selected_urls = self.product_urls
        else:
            try:
                count = int(import_count)
                if count <= 0:
                    print("⚠️  Invalid count. Extracting only the sample product.")
                    selected_urls = sample_urls
                elif count >= len(self.product_urls):
                    selected_urls = self.product_urls
                else:
                    # Select evenly distributed products
                    step = len(self.product_urls) // count
                    selected_urls = self.product_urls[::step][:count]
            except ValueError:
                print("⚠️  Invalid input. Extracting only the sample product.")
                selected_urls = sample_urls
        
        # Step 4: Extract selected products
        if len(selected_urls) > 1:
            print(f"\n📥 Extracting {len(selected_urls)} products...")
            # Extract all selected products (excluding the sample which is already extracted)
            urls_to_extract = [url for url in selected_urls if url not in sample_urls]
            if urls_to_extract:
                additional_extracted = self._extract_products_batch(urls_to_extract, custom_instructions)
                # Combine sample with additional products
                extracted_samples.extend(additional_extracted)
        else:
            print(f"\n✅ Using sample product only.")
        
        self.extracted_products = extracted_samples
        
        # Convert to standardized output format
        output = self._format_output(extracted_samples)
        
        # Validate output
        is_valid, error = validate_legacy_site_output(output)
        if not is_valid:
            self.logger.warning(f"Output validation warning: {error}")
        
        # Save output
        save_state("legacy_site_extraction", output)
        self.logger.info(f"Extraction complete. Extracted {len(output['products'])} products")
        
        return output
    
    def _iterative_refinement_loop(
        self,
        extracted_samples: List[Dict[str, Any]],
        sample_urls: List[str],
        custom_instructions: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Handle iterative refinement loop for extraction."""
        iteration = 1
        iteration_history = []
        
        while True:
            # Save current state
            save_state("extraction", {
                "sample_size": len(extracted_samples),
                "extracted_products": extracted_samples,
                "custom_instructions_used": custom_instructions is not None,
                "iteration": iteration,
                "iteration_history": iteration_history
            })
            
            # Display sample for user review
            if not extracted_samples:
                print("\n⚠️  No products extracted. Exiting iteration loop.")
                break
            
            sample = extracted_samples[0]
            print("\n" + "="*60)
            print(f"📊 ITERATION {iteration} - SAMPLE PRODUCT MAPPING")
            print("="*60)
            print(json.dumps(sample["mapped_data"], indent=2, ensure_ascii=False))
            
            print("\n" + "="*60)
            print("🔄 EXTRACTION ITERATION")
            print("="*60)
            print("\nOptions:")
            print("  - Type 'done' to accept and proceed")
            print("  - Type 'refine' to edit custom prompt and re-extract")
            print("  - Type 'retry' to re-extract with current prompt")
            print("  - Type 'feedback' to provide corrections (will be added to prompt)")
            print("  - Press Enter to continue")
            
            user_action = input("\nWhat would you like to do? ").strip().lower()
            
            if user_action == "done":
                print("\n✅ Extraction approved. Proceeding with current results.")
                iteration_history.append({
                    "iteration": iteration,
                    "action": "approved",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                break
            
            elif user_action == "refine":
                print("\n⚙️  Refining extraction prompt...")
                new_instructions = edit_custom_prompt_interactive()
                if new_instructions:
                    custom_instructions = new_instructions
                    iteration_history.append({
                        "iteration": iteration,
                        "action": "prompt_refined",
                        "custom_instructions_length": len(custom_instructions)
                    })
                print("\n🔄 Re-extracting with updated prompt...")
                extracted_samples = self._extract_products_batch(sample_urls, custom_instructions)
                iteration += 1
            
            elif user_action == "retry":
                print("\n🔄 Re-extracting with current prompt...")
                extracted_samples = self._extract_products_batch(sample_urls, custom_instructions)
                iteration_history.append({
                    "iteration": iteration,
                    "action": "retried",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                iteration += 1
            
            elif user_action == "feedback":
                print("\n💬 Please provide your feedback/corrections:")
                feedback = input("Feedback: ").strip()
                
                if feedback:
                    if custom_instructions:
                        custom_instructions = f"{custom_instructions}\n\nUser Feedback (Iteration {iteration}): {feedback}"
                    else:
                        custom_instructions = f"User Feedback (Iteration {iteration}): {feedback}"
                    
                    save_custom_prompt(custom_instructions)
                    
                    iteration_history.append({
                        "iteration": iteration,
                        "action": "feedback_provided",
                        "feedback": feedback
                    })
                    
                    print("\n✅ Feedback added to prompt. Re-extracting...")
                    extracted_samples = self._extract_products_batch(sample_urls, custom_instructions)
                    iteration += 1
            
            else:
                print("\n✅ Proceeding with current extraction results.")
                break
        
        return extracted_samples
    
    def _extract_products_batch(
        self,
        urls: List[str],
        custom_instructions: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract a batch of products from URLs with proper rate limiting."""
        extracted_samples = []
        
        print(f"\n📥 Extracting {len(urls)} product(s)...")
        self.logger.info(f"Extracting {len(urls)} products")
        
        # Initial delay before first LLM call to avoid immediate rate limits
        time.sleep(0.5)
        
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        for i, url in enumerate(urls, 1):
            print(f"\n   [{i}/{len(urls)}] Processing: {url}")
            self.logger.debug(f"Processing URL {i}/{len(urls)}: {url}")
            
            # Add delay before LLM call (increases if we've had errors)
            if consecutive_errors > 0:
                delay = 2.0 * (2 ** min(consecutive_errors - 1, 3))  # Exponential backoff: 2s, 4s, 8s
                print(f"     ⏳ Waiting {delay:.1f}s before processing (due to previous errors)...")
                time.sleep(delay)
            elif i > 1:
                # Normal delay between calls to avoid rate limits
                time.sleep(2.0)
            
            try:
                # Fetch HTML
                r = self.session.get(url, timeout=30)
                if r.status_code != 200:
                    print(f"     ⚠️  HTTP {r.status_code}, skipping...")
                    self.logger.warning(f"HTTP {r.status_code} for URL: {url}")
                    continue
                
                html_content = r.text
                
                # Use Gemini to map to VTEX schema (has built-in retry logic)
                # LLM will extract images, categories, and all product data from HTML
                print(f"     🤖 Extracting product data with Gemini (images, categories, and all fields)...")
                
                mapped_data = extract_to_vtex_schema(
                    html_content,
                    url,
                    self.gemini_api_key,
                    custom_instructions=custom_instructions
                )
                
                # Reset error counter on success
                consecutive_errors = 0
                
                # LLM has extracted everything, including images and categories
                images = mapped_data.get("images", [])
                categories = mapped_data.get("categories", [])
                
                print(f"     🖼️  LLM identified {len(images)} product images")
                print(f"     📂 LLM extracted {len(categories)} categories")
                
                # Store extracted data (LLM has already extracted images and categories)
                extracted_samples.append({
                    "url": url,
                    "html_preview": html_content[:1000],
                    "images": images,  # Use LLM-extracted images only
                    "mapped_data": mapped_data
                })
                
                print(f"     ✅ Extraction complete")
                self.logger.debug(f"Successfully extracted product from: {url}")
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a rate limit error
                is_rate_limit = (
                    "429" in error_msg or
                    "resource exhausted" in error_msg.lower() or
                    "rate limit" in error_msg.lower()
                )
                
                if is_rate_limit:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"     ⚠️  Rate limit error (429). Too many consecutive errors.")
                        print(f"     💡 Tip: Wait a few minutes and retry, or check your API quota.")
                        print(f"     ⏸️  Pausing extraction. Processed {len(extracted_samples)}/{len(urls)} products so far.")
                        break
                    else:
                        # The retry logic in extract_to_vtex_schema should handle this,
                        # but if it still fails, we'll wait longer before next product
                        print(f"     ⚠️  Rate limit error detected. Will wait longer before next product.")
                else:
                    # Non-rate-limit errors: log and continue
                    print(f"     ⚠️  Error extracting {url}: {type(e).__name__}")
                    self.logger.error(f"Error extracting {url}: {e}", exc_info=True)
                    # Reset consecutive errors for non-rate-limit errors
                    consecutive_errors = 0
                    continue
        
        return extracted_samples
    
    def _format_output(self, extracted_products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format extracted products into standardized output JSON.
        
        Args:
            extracted_products: List of extracted product data
            
        Returns:
            Standardized output dictionary
        """
        products = []
        
        for product_data in extracted_products:
            mapped = product_data.get("mapped_data", {})
            
            # Extract product info
            product_info = mapped.get("product", {})
            categories = mapped.get("categories", [])
            brand = mapped.get("brand", {})
            skus = mapped.get("skus", [])
            specifications = mapped.get("specifications", [])
            images = product_data.get("images", [])
            
            # Format product entry (include mapped_data for backward compatibility)
            product_entry = {
                "url": product_data.get("url"),
                "product": product_info,
                "categories": categories,
                "brand": brand,
                "skus": skus,
                "specifications": specifications,
                "images": images,
                "mapped_data": mapped  # Include for reporting phase compatibility
            }
            
            products.append(product_entry)
        
        return {
            "target_url": self.target_url,
            "extracted_at": datetime.now().isoformat(),
            "products": products,
            "metadata": {
                "total_products": len(products),
                "total_urls_found": len(self.product_urls),
                "custom_prompt_used": get_custom_prompt() is not None
            }
        }
    
    def _review_urls_with_llm(self, urls: List[str]) -> List[str]:
        """
        Review ALL URLs with LLM to identify which ones are Product Detail Pages (PDPs).
        Processes URLs in batches and applies LLM categorization to the full list.
        Allows iterative refinement based on LLM analysis and user feedback.
        
        Args:
            urls: List of ALL URLs found by crawler (not pre-filtered)
            
        Returns:
            Filtered list of product URLs (PDPs only)
        """
        if not urls:
            return urls
        
        if not self.gemini_api_key:
            print("   ⚠️  GEMINI_API_KEY not found. Skipping LLM review.")
            return urls
        
        print(f"\n   🤖 LLM will analyze {len(urls)} URLs to identify Product Detail Pages (PDPs)")
        
        iteration = 1
        all_pdp_urls = set()
        batch_size = 100  # Process URLs in batches
        
        # Process URLs in batches
        for batch_start in range(0, len(urls), batch_size):
            batch_end = min(batch_start + batch_size, len(urls))
            batch_urls = urls[batch_start:batch_end]
            batch_num = (batch_start // batch_size) + 1
            total_batches = (len(urls) + batch_size - 1) // batch_size
            
            print(f"\n   📦 Processing batch {batch_num}/{total_batches} ({len(batch_urls)} URLs)")
            
            batch_retries = 0
            max_batch_retries = 3
            
            while batch_retries < max_batch_retries:
                try:
                    # Small delay before LLM call to avoid rate limits
                    if batch_retries > 0:
                        wait_time = 2.0 * (2 ** batch_retries)  # Exponential backoff
                        print(f"   ⏳ Waiting {wait_time:.1f}s before retry...")
                        time.sleep(wait_time)
                    elif batch_num > 1:
                        time.sleep(1.0)  # Small delay between batches
                    
                    model = initialize_gemini(self.gemini_api_key)
                    
                    prompt = f"""You are analyzing URLs from a website to identify Product Detail Pages (PDPs).

Target Website: {self.target_url}
Batch: {batch_num}/{total_batches}
Total URLs to analyze: {len(batch_urls)}

URLs to analyze:
{json.dumps(batch_urls, indent=2, ensure_ascii=False)}

For EACH URL in the list above, categorize it as:
1. "definitely_pdp" - Clearly a Product Detail Page (has product ID, product path, etc.)
2. "possibly_pdp" - Might be a PDP but needs verification
3. "not_pdp" - Clearly NOT a PDP (home page, category page, about page, contact, etc.)

Consider:
- URL patterns (product IDs like /12345/, /product-123/, /p/slug, etc.)
- URL structure and depth
- Common PDP indicators: product IDs, /product/, /p/, /item/, /produto/
- Common non-PDP indicators: /category/, /categoria/, /about/, /contact/, /home/, /index, etc.

Return a JSON object with this structure:
{{
    "definitely_pdp": ["list of ALL URLs from the input that are definitely product pages"],
    "possibly_pdp": ["list of ALL URLs from the input that might be product pages"],
    "not_pdp": ["list of ALL URLs from the input that are clearly not product pages"],
    "patterns": {{
        "pdp_patterns": ["regex patterns or URL structures that indicate PDPs"],
        "non_pdp_patterns": ["regex patterns or URL structures that indicate non-PDPs"]
    }},
    "analysis": "Brief explanation of your categorization logic"
}}

IMPORTANT: You must categorize EVERY URL from the input list. Return valid JSON only."""

                    # Call Gemini with retry logic
                    def _call_gemini_api():
                        if isinstance(model, tuple):
                            # New SDK
                            client, model_name = model
                            response = client.models.generate_content(
                                model=model_name,
                                contents=prompt
                            )
                            # New SDK response structure may differ - check for text attribute or candidates
                            if hasattr(response, 'text'):
                                return response.text.strip()
                            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                                candidate = response.candidates[0]
                                if hasattr(candidate, 'content'):
                                    if hasattr(candidate.content, 'parts'):
                                        return ''.join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                                    elif hasattr(candidate.content, 'text'):
                                        return candidate.content.text.strip()
                            return str(response)
                        else:
                            # Legacy SDK
                            response = model.generate_content(prompt)
                            return response.text.strip()
                    
                    analysis_text = _retry_with_exponential_backoff(_call_gemini_api)
                    
                    # Parse response
                    analysis_text = analysis_text.strip()
                    # Remove markdown code blocks if present
                    if analysis_text.startswith("```"):
                        analysis_text = analysis_text.split("```")[1]
                        if analysis_text.startswith("json"):
                            analysis_text = analysis_text[4:]
                    analysis_text = analysis_text.strip()
                    
                    analysis = json.loads(analysis_text)
                    
                    definitely_pdp = set(analysis.get("definitely_pdp", []))
                    possibly_pdp = set(analysis.get("possibly_pdp", []))
                    not_pdp = set(analysis.get("not_pdp", []))
                    
                    # Add definitely_pdp and possibly_pdp to our collection
                    all_pdp_urls.update(definitely_pdp)
                    all_pdp_urls.update(possibly_pdp)
                    
                    print(f"      ✅ Definitely PDP: {len(definitely_pdp)}")
                    print(f"      ⚠️  Possibly PDP: {len(possibly_pdp)}")
                    print(f"      ❌ Not PDP: {len(not_pdp)}")
                    
                    # Successfully processed batch
                    break
                    
                except json.JSONDecodeError as e:
                    batch_retries += 1
                    if batch_retries < max_batch_retries:
                        print(f"      ⚠️  Error parsing LLM response. Retrying batch {batch_num}...")
                        continue
                    else:
                        print(f"      ⚠️  Max retries reached for batch {batch_num}. Skipping batch.")
                        break
                        
                except Exception as e:
                    error_msg = str(e)
                    is_rate_limit = (
                        "429" in error_msg or
                        "resource exhausted" in error_msg.lower() or
                        "rate limit" in error_msg.lower()
                    )
                    
                    if is_rate_limit:
                        batch_retries += 1
                        if batch_retries < max_batch_retries:
                            wait_time = 5.0 * (2 ** batch_retries)
                            print(f"      ⚠️  Rate limit error. Waiting {wait_time:.1f}s before retry...")
                            time.sleep(wait_time)
                            continue
                        else:
                            print(f"      ⚠️  Rate limit error persisted. Skipping batch {batch_num}.")
                            print(f"      💡 Tip: Wait a few minutes and retry, or check your API quota.")
                            break
                    else:
                        print(f"      ⚠️  Error processing batch {batch_num}. Skipping batch.")
                        self.logger.error(f"Error in LLM URL review batch {batch_num}: {e}", exc_info=True)
                        break
        
        # Convert to list and deduplicate
        final_pdp_urls = list(all_pdp_urls)
        
        print(f"\n   ✅ LLM Review Complete:")
        print(f"      Total URLs analyzed: {len(urls)}")
        print(f"      PDPs identified: {len(final_pdp_urls)}")
        
        if final_pdp_urls:
            print(f"\n   📋 Sample PDPs identified:")
            for url in final_pdp_urls[:10]:
                print(f"      • {url}")
            if len(final_pdp_urls) > 10:
                print(f"      ... and {len(final_pdp_urls) - 10} more")
        
        # Ask user for feedback (optional refinement)
        print("\n" + "="*60)
        print("🔄 URL REVIEW - OPTIONAL REFINEMENT")
        print("="*60)
        print(f"\nLLM identified {len(final_pdp_urls)} PDPs from {len(urls)} total URLs.")
        print("\nOptions:")
        print("  - Press Enter to accept and proceed")
        print("  - Type 'include <url>' to explicitly add a URL")
        print("  - Type 'exclude <url>' to explicitly remove a URL")
        print("  - Type 'show' to see all identified PDPs")
        print("  - Type 'retry' to re-analyze all URLs with LLM")
        
        user_action = input("\nWhat would you like to do? ").strip().lower()
        
        if user_action == "retry":
            # Recursively retry the entire process
            return self._review_urls_with_llm(urls)
        
        elif user_action == "show":
            print(f"\n📋 All identified PDPs ({len(final_pdp_urls)}):")
            for i, url in enumerate(final_pdp_urls, 1):
                print(f"  {i}. {url}")
            # Ask again after showing
            user_action = input("\nPress Enter to proceed or type a command: ").strip().lower()
        
        if user_action.startswith("include "):
            url_to_include = user_action[8:].strip()
            if url_to_include not in final_pdp_urls:
                final_pdp_urls.append(url_to_include)
                print(f"✅ Added: {url_to_include}")
            else:
                print(f"ℹ️  URL already in list: {url_to_include}")
        
        elif user_action.startswith("exclude "):
            url_to_exclude = user_action[8:].strip()
            if url_to_exclude in final_pdp_urls:
                final_pdp_urls.remove(url_to_exclude)
                print(f"❌ Removed: {url_to_exclude}")
            else:
                print(f"ℹ️  URL not found in list: {url_to_exclude}")
        
        print(f"\n✅ Proceeding with {len(final_pdp_urls)} PDP URLs.")
        return final_pdp_urls
    
    def extract_all_products(self, custom_instructions: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract all products without iterative refinement (for bulk extraction).
        
        Args:
            custom_instructions: Optional custom extraction instructions
            
        Returns:
            Dictionary with all extracted products
        """
        self.logger.info("Starting bulk extraction of all products")
        return self.extract_products(
            sample_size=len(self.product_urls),
            custom_instructions=custom_instructions,
            enable_iterative_refinement=False
        )

