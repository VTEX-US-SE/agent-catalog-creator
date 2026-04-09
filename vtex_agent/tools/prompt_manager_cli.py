#!/usr/bin/env python3
"""Command-line utility to manage custom extraction prompts."""
import sys
import os
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from vtex_agent.utils.prompt_manager import (
    get_custom_prompt,
    set_custom_prompt,
    clear_custom_prompt,
    edit_custom_prompt_interactive
)


def main():
    """Main CLI interface for prompt management."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m vtex_agent.tools.prompt_manager_cli show          - Show current custom prompt")
        print("  python -m vtex_agent.tools.prompt_manager_cli edit          - Edit custom prompt interactively")
        print("  python -m vtex_agent.tools.prompt_manager_cli set <text>    - Set custom prompt from command line")
        print("  python -m vtex_agent.tools.prompt_manager_cli clear          - Clear custom prompt")
        print("  python -m vtex_agent.tools.prompt_manager_cli file <path>   - Load prompt from file")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "show":
        prompt = get_custom_prompt()
        if prompt:
            print("\n📝 Current custom extraction prompt:")
            print("=" * 60)
            print(prompt)
            print("=" * 60)
            print(f"\nLength: {len(prompt)} characters")
        else:
            print("\n⚠️  No custom prompt configured. Using default extraction prompt.")
    
    elif command == "edit":
        edit_custom_prompt_interactive()
    
    elif command == "set":
        if len(sys.argv) < 3:
            print("❌ Error: Please provide the prompt text")
            print("   Usage: python -m vtex_agent.tools.prompt_manager_cli set 'Your custom instructions here'")
            sys.exit(1)
        prompt_text = " ".join(sys.argv[2:])
        set_custom_prompt(prompt_text)
        print(f"✅ Custom prompt saved ({len(prompt_text)} characters)")
    
    elif command == "clear":
        clear_custom_prompt()
        print("✅ Custom prompt cleared")
    
    elif command == "file":
        if len(sys.argv) < 3:
            print("❌ Error: Please provide the file path")
            print("   Usage: python -m vtex_agent.tools.prompt_manager_cli file /path/to/prompt.txt")
            sys.exit(1)
        file_path = sys.argv[2]
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
            set_custom_prompt(prompt_text)
            print(f"✅ Custom prompt loaded from file ({len(prompt_text)} characters)")
        except FileNotFoundError:
            print(f"❌ Error: File not found: {file_path}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            sys.exit(1)
    
    else:
        print(f"❌ Unknown command: {command}")
        print("   Use 'show', 'edit', 'set', 'clear', or 'file'")
        sys.exit(1)


if __name__ == "__main__":
    main()

