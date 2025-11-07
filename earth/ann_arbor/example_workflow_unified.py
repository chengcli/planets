#!/usr/bin/env python3
"""
Example: Complete Ann Arbor Weather Data Pipeline (Unified System)

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
    print("Ann Arbor Weather Data Pipeline - Unified System Example")
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
- Test area: Ann Arbor, Michigan (42.3°N, 83.7°W)
- Domain: 125 km × 125 km horizontal, 15 km vertical
- Time: November 1, 2025
- Grid: 150 × 200 × 200 interior cells + 3 ghost cells
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
    print("  python generate_config.py ann-arbor --output custom_ann_arbor.yaml")
    print()
    print("Or with custom parameters:")
    print("  python generate_config.py ann-arbor \\")
    print("    --start-date 2025-11-15 \\")
    print("    --end-date 2025-11-17 \\")
    print("    --nx2 300 --nx3 300 \\")
    print("    --output ann_arbor_highres.yaml")
    print()
    print("Note: Default configs already exist in location directories")
    print()
    
    # Step 2: Download and Process Data
    print_step(2, "Download and Process Data (Complete Pipeline)")
    print("Run the complete pipeline in one command:")
    print()
    print("Command:")
    print("  python download_location_data.py ann-arbor")
    print()
    print("This automatically executes all 4 steps:")
    print("  1. Fetch ERA5 data from ECMWF")
    print("  2. Calculate air density")
    print("  3. Regrid to Cartesian coordinates")
    print("  4. Compute hydrostatic pressure")
    print()
    print("With custom config:")
    print("  python download_location_data.py ann-arbor \\")
    print("    --config custom_ann_arbor.yaml")
    print()
    print("Stop after specific step:")
    print("  python download_location_data.py ann-arbor --stop-after 2")
    print()
    print("Time: Typically 20-60 minutes depending on network")
    print()
    
    # Advanced Examples
    print_step(3, "Advanced Examples")
    print()
    print("High-resolution simulation:")
    print("  python generate_config.py ann-arbor \\")
    print("    --nx1 200 --nx2 400 --nx3 400 \\")
    print("    --output ann_arbor_highres.yaml")
    print("  python download_location_data.py ann-arbor \\")
    print("    --config ann_arbor_highres.yaml \\")
    print("    --timeout 7200")
    print()
    print("Multi-day simulation:")
    print("  python generate_config.py ann-arbor \\")
    print("    --start-date 2025-11-01 \\")
    print("    --end-date 2025-11-03 \\")
    print("    --tlim 172800 \\")
    print("    --output ann_arbor_3day.yaml")
    print("  python download_location_data.py ann-arbor \\")
    print("    --config ann_arbor_3day.yaml")
    print()
    print("Custom output directory:")
    print("  python download_location_data.py ann-arbor \\")
    print("    --output-base ./november_2025_data")
    print()
    
    # Backward Compatibility
    print_step(4, "Backward Compatibility")
    print("Existing workflows still work:")
    print()
    print("Old way (still supported):")
    print("  cd ann_arbor")
    print("  python download_ann_arbor_data.py")
    print()
    print("New way (recommended):")
    print("  cd earth")
    print("  python download_location_data.py ann-arbor")
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
    print("  python ../generate_config.py ann-arbor [options]")
    print()
    print("Download data:")
    print("  python ../download_location_data.py ann-arbor [options]")
    print()
    print("Get help:")
    print("  python ../generate_config.py --help")
    print("  python ../download_location_data.py --help")
    print()


if __name__ == "__main__":
    main()
