"""State management for persistent workflow execution."""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

STATE_DIR = Path(__file__).parent.parent.parent / "state"

# Mapping of step names to their order in the workflow
STEP_ORDER = {
    "discovery": 1,
    "mapping": 2,
    "extraction": 3,
    "legacy_site_extraction": None,  # No number prefix (final output file)
    "sampling": 5,
    "reporting": 6,
    "vtex_category_tree": 7,
    "vtex_products_skus": 8,
    "vtex_images": 9,
    "execution": 10,
    "vtex_specifications": 11,
    "field_type_overrides": 12,
    "custom_prompt": None,  # No number prefix (not part of main workflow)
}


def ensure_state_dir():
    """Ensure state directory exists."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def save_state(step_name: str, data: Dict[str, Any]) -> str:
    """
    Save state to a JSON file with numeric prefix based on workflow order.
    
    Args:
        step_name: Name of the step (e.g., 'discovery', 'mapping', 'extraction')
        data: State data to persist
        
    Returns:
        Path to saved state file
    """
    ensure_state_dir()
    
    # Get order number for this step
    order = STEP_ORDER.get(step_name)
    
    # Build filename with numeric prefix if order exists
    if order is not None:
        filename = f"{order:02d}_{step_name}.json"
    else:
        filename = f"{step_name}.json"
    
    state_file = STATE_DIR / filename
    
    # Remove old unnumbered file if it exists (for migration)
    old_file = STATE_DIR / f"{step_name}.json"
    if old_file.exists() and old_file != state_file:
        try:
            old_file.unlink()
        except Exception:
            pass  # Ignore errors when removing old file
    
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return str(state_file)


def load_state(step_name: str) -> Optional[Dict[str, Any]]:
    """
    Load state from a JSON file.
    Tries numbered filename first, then falls back to unnumbered for backward compatibility.
    
    Args:
        step_name: Name of the step to load
    
    Returns:
        State data or None if file doesn't exist
    """
    # Get order number for this step
    order = STEP_ORDER.get(step_name)
    
    # Try numbered filename first
    if order is not None:
        state_file = STATE_DIR / f"{order:02d}_{step_name}.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)
    
    # Fallback to unnumbered filename (for backward compatibility)
    state_file = STATE_DIR / f"{step_name}.json"
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    return None


def get_state_path(step_name: str) -> str:
    """
    Get the path to a state file without loading it.
    Returns numbered filename if order exists, otherwise unnumbered.
    """
    order = STEP_ORDER.get(step_name)
    if order is not None:
        return str(STATE_DIR / f"{order:02d}_{step_name}.json")
    return str(STATE_DIR / f"{step_name}.json")


def save_custom_prompt(instructions: str) -> str:
    """
    Save custom extraction prompt instructions to state.
    
    Args:
        instructions: Custom instructions to append to extraction prompts
        
    Returns:
        Path to saved state file
    """
    ensure_state_dir()
    state_file = STATE_DIR / "custom_prompt.json"
    data = {
        "instructions": instructions,
        "updated_at": str(Path(__file__).stat().st_mtime) if Path(__file__).exists() else None
    }
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return str(state_file)


def load_custom_prompt() -> Optional[str]:
    """
    Load custom extraction prompt instructions from state.
    
    Returns:
        Custom instructions string or None if not found
    """
    state_file = STATE_DIR / "custom_prompt.json"
    if not state_file.exists():
        return None
    with open(state_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("instructions")

