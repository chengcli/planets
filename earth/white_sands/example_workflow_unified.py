#!/usr/bin/env python3
"""
Example: Complete White Sands Weather Data Pipeline (Unified System)

This example demonstrates the complete workflow using the new unified
location configuration system.

The unified system provides:
- Single download script for all locations
- Easy configuration generation
- Flexible command-line overrides
- Reduced code duplication

Author: Earth Weather Pipeline Team
Date: 2025
"""

def print_step(step_num, title):
    """Print a formatted step header."""
    print("\n" + "="*70)
    print(f"STEP {step_num}: {title}")
    print("="*70 + "\n")


def main():
    print("\n" + "="*70)
    print("White Sands Weather Data Pipeline - Unified System Example")
    print("="*70)
    
    print("""
This example demonstrates the complete pipeline using the new unified
location configuration system.

Benefits of the Unified System:
- Single download script works for all locations
- Easy to add new locations without code changes
- Flexible configuration via command-line arguments
- Consistent interface across all locations
- Reduced code duplication

The pipeline covers:
- Test area: 106.7°W - 106.2°W, 32.6°N - 33.6°N
- With buffer: 107.2°W - 105.7°W, 32.1°N - 34.1°N
- Time: October 1-2, 2025
- Domain: ~223 km N-S × ~140 km E-W × 15 km vertical
""")
    
    # Overview
    print_step("OVERVIEW", "Unified System Components")
    print("The unified system consists of:")
    print()
    print("1. Location Table (locations.yaml)")
    print("   - Defines all available locations")
    print("   - Specifies geographic bounds, center, and defaults")
    print("   - Easy to add new locations")
    print()
    print("2. Configuration Template (config_template.yaml)")
    print("   - Template YAML with placeholders")
    print("   - Same structure for all locations")
    print()
    print("3. Configuration Generator (generate_config.py)")
    print("   - Generates location-specific configs")
    print("   - Supports command-line overrides")
    print()
    print("4. Unified Download Script (download_location_data.py)")
    print("   - Single script for all locations")
    print("   - Replaces per-location download scripts")
    print()
    
    # Step 0: List Available Locations
    print_step(0, "List Available Locations")
    print("Command:")
    print("  cd /path/to/planets/earth")
    print("  python generate_config.py --list")
    print()
    print("This shows all configured locations:")
    print("  - ann-arbor: Ann Arbor, Michigan")
    print("  - white-sands: White Sands, New Mexico")
    print("  - (more can be added to locations.yaml)")
    print()
    
    # Step 1: Generate Configuration (Optional)
    print_step(1, "Generate Configuration File (Optional)")
    print("If you need a custom configuration:")
    print()
    print("Command:")
    print("  python generate_config.py white-sands --output custom_white_sands.yaml")
    print()
    print("Or with custom parameters:")
    print("  python generate_config.py white-sands \\")
    print("    --start-date 2025-10-15 \\")
    print("    --end-date 2025-10-20 \\")
    print("    --nx2 500 --nx3 400 \\")
    print("    --output white_sands_highres.yaml")
    print()
    print("Note: Default configs already exist in location directories")
    print()
    
    # Step 2: Download and Process Data
    print_step(2, "Download and Process Data (Complete Pipeline)")
    print("Run the complete pipeline in one command:")
    print()
    print("Command:")
    print("  python download_location_data.py white-sands")
    print()
    print("This automatically executes all 4 steps:")
    print("  1. Fetch ERA5 data from ECMWF")
    print("  2. Calculate air density")
    print("  3. Regrid to Cartesian coordinates")
    print("  4. Compute hydrostatic pressure")
    print()
    print("With custom config:")
    print("  python download_location_data.py white-sands \\")
    print("    --config custom_white_sands.yaml")
    print()
    print("Stop after specific step:")
    print("  python download_location_data.py white-sands --stop-after 2")
    print()
    print("Time: Typically 20-60 minutes depending on network")
    print()
    
    # Advanced Examples
    print_step(3, "Advanced Examples")
    print()
    print("High-resolution simulation:")
    print("  python generate_config.py white-sands \\")
    print("    --nx1 200 --nx2 500 --nx3 400 \\")
    print("    --output white_sands_highres.yaml")
    print("  python download_location_data.py white-sands \\")
    print("    --config white_sands_highres.yaml \\")
    print("    --timeout 7200")
    print()
    print("Extended time period:")
    print("  python generate_config.py white-sands \\")
    print("    --start-date 2025-10-01 \\")
    print("    --end-date 2025-10-07 \\")
    print("    --tlim 604800 \\")
    print("    --output white_sands_week.yaml")
    print("  python download_location_data.py white-sands \\")
    print("    --config white_sands_week.yaml")
    print()
    print("Custom output directory:")
    print("  python download_location_data.py white-sands \\")
    print("    --output-base ./october_2025_data")
    print()
    
    # Backward Compatibility
    print_step(4, "Backward Compatibility")
    print("Existing workflows still work:")
    print()
    print("Old way (still supported):")
    print("  cd white_sands")
    print("  python download_white_sands_data.py")
    print()
    print("New way (recommended):")
    print("  cd earth")
    print("  python download_location_data.py white-sands")
    print()
    print("Both produce the same results.")
    print()
    
    # Adding New Locations
    print_step(5, "Adding New Locations")
    print("To add a new location, edit locations.yaml:")
    print()
    print("1. Add entry under 'locations:' with unique ID")
    print("2. Define polygon bounds (vertices)")
    print("3. Set center coordinates and defaults")
    print("4. Generate config: python generate_config.py <new-id>")
    print("5. Download data: python download_location_data.py <new-id>")
    print()
    print("No code changes needed!")
    print()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70 + "\n")
    print("The unified system provides:")
    print()
    print("✓ Single download script for all locations")
    print("✓ Easy configuration generation with overrides")
    print("✓ Add new locations without code changes")
    print("✓ Backward compatible with existing workflows")
    print("✓ Reduced code duplication (DRY principle)")
    print("✓ Consistent interface across locations")
    print()
    print("For detailed documentation, see:")
    print("  - ../README_UNIFIED_SYSTEM.md")
    print("  - README.md (this directory)")
    print("  - ../locations.yaml (location definitions)")
    print()
    
    # Quick Reference
    print("="*70)
    print("QUICK REFERENCE")
    print("="*70 + "\n")
    print("List locations:")
    print("  python ../generate_config.py --list")
    print()
    print("Generate config:")
    print("  python ../generate_config.py white-sands [options]")
    print()
    print("Download data:")
    print("  python ../download_location_data.py white-sands [options]")
    print()
    print("Get help:")
    print("  python ../generate_config.py --help")
    print("  python ../download_location_data.py --help")
    print()


if __name__ == "__main__":
    main()
