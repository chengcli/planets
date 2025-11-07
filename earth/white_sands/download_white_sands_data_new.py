#!/usr/bin/env python3
"""
White Sands Weather Data Download Script (Compatibility Wrapper)

This is a backward-compatibility wrapper that calls the unified
download_location_data.py script with the white-sands location.

For new workflows, please use the unified script directly:
    python ../download_location_data.py white-sands [options]

Usage:
    python download_white_sands_data.py [options]

Options are passed through to the unified script.
"""

import sys
import subprocess
from pathlib import Path

def main():
    # Get the parent directory (earth)
    script_dir = Path(__file__).parent
    earth_dir = script_dir.parent
    
    # Path to unified script
    unified_script = earth_dir / "download_location_data.py"
    
    if not unified_script.exists():
        print(f"ERROR: Unified script not found: {unified_script}")
        return 1
    
    # Build command with location ID
    command = ["python3", str(unified_script), "white-sands"]
    
    # Add all command-line arguments (skip script name)
    command.extend(sys.argv[1:])
    
    # If no --config specified, add default config path
    if "--config" not in sys.argv:
        config_path = script_dir / "white_sands.yaml"
        if config_path.exists():
            command.extend(["--config", str(config_path)])
    
    # Execute unified script
    try:
        result = subprocess.run(command, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
