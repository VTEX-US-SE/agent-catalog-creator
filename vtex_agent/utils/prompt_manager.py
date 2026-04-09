"""Utility functions for managing custom extraction prompts."""
import os
from typing import Optional
from .state_manager import save_custom_prompt, load_custom_prompt


def set_custom_prompt(instructions: str) -> str:
    """
    Set custom extraction prompt instructions.
    
    Args:
        instructions: Custom instructions to optimize extraction
        
    Returns:
        Path to saved state file
    """
    return save_custom_prompt(instructions)


def get_custom_prompt() -> Optional[str]:
    """
    Get current custom extraction prompt instructions.
    
    Returns:
        Custom instructions string or None if not set
    """
    return load_custom_prompt()


def clear_custom_prompt() -> str:
    """
    Clear custom extraction prompt instructions.
    
    Returns:
        Path to saved state file
    """
    return save_custom_prompt("")


def edit_custom_prompt_interactive():
    """
    Interactive function to edit custom prompt instructions.
    
    Returns:
        Custom instructions string or None if cleared
    """
    existing = get_custom_prompt()
    
    if existing:
        print("\n📝 Current custom instructions:")
        print("-" * 60)
        print(existing)
        print("-" * 60)
        action = input("\n   (e)dit, (c)lear, or (k)eep? [e/c/k]: ").strip().lower()
        
        if action == "c":
            clear_custom_prompt()
            print("✅ Custom instructions cleared")
            return None
        elif action == "k":
            return existing
    
    print("\n💡 Enter custom instructions to optimize product/SKU extraction.")
    print("   These will take PRIORITY over default extraction rules.")
    print("\n   Examples (CSS selector-based extraction):")
    print("   - Product ID and SKU ID = <span class='code'>10010801</span>")
    print("   - Product Description = text from 'body > main > section.product-details > span.description'")
    print("   - Product Specifications = all <dd> key-value pairs inside 'body > main > dl.product-classification'")
    print("   - SKU EAN = <span class='code'>10010801</span>")
    print("   - SKU Price = 'value' attribute from input '#qty-[ProductID]'")
    print("   - Image = src from 'body > main > section > div.carousel > img'")
    print("\n   Format:")
    print("   - Use CSS selectors: 'body > main > section.class-name'")
    print("   - Specify attributes: 'src from img', 'value from input', 'text from span'")
    print("   - Use element syntax: <span class='code'> for element matching")
    print("\n   (Press Enter twice or type 'done' on a new line to finish)")
    print("   (Type 'clear' to remove custom instructions)")
    
    instructions_lines = []
    while True:
        try:
            line = input()
            if line.strip().lower() == "done":
                break
            if line.strip().lower() == "clear":
                clear_custom_prompt()
                print("✅ Custom instructions cleared")
                return None
            if line == "" and instructions_lines and instructions_lines[-1] == "":
                break
            instructions_lines.append(line)
        except (EOFError, KeyboardInterrupt):
            if instructions_lines:
                break
            return None
    
    instructions = "\n".join(instructions_lines).strip()
    
    if instructions:
        set_custom_prompt(instructions)
        print(f"\n✅ Custom instructions saved ({len(instructions)} characters)")
        return instructions
    else:
        print("\n⚠️  No instructions provided")
        return None

