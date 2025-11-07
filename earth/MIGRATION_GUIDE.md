# Migration Guide: From Location-Specific to Unified System

This guide helps you migrate from the old location-specific scripts to the new unified location configuration system.

## What Changed?

The new unified system replaces location-specific duplicate scripts with a single, flexible system:

**Old System (Before):**
- Separate `download_ann_arbor_data.py` script (~449 lines)
- Separate `download_white_sands_data.py` script (~449 lines)
- Duplicate code in both scripts (>90% identical)
- Adding new locations requires copying and modifying scripts

**New System (After):**
- Single `download_location_data.py` script (works for all locations)
- Location data in `locations.yaml` file
- Configuration generator `generate_config.py`
- Adding new locations requires only editing YAML

## Quick Migration

### For Ann Arbor Users

**Old workflow:**
```bash
cd earth/ann_arbor
python download_ann_arbor_data.py
```

**New workflow (Option 1 - Recommended):**
```bash
cd earth
python download_location_data.py ann-arbor
```

**New workflow (Option 2 - Using wrapper):**
```bash
cd earth/ann_arbor
python download_ann_arbor_data_new.py
```

### For White Sands Users

**Old workflow:**
```bash
cd earth/white_sands
python download_white_sands_data.py
```

**New workflow (Option 1 - Recommended):**
```bash
cd earth
python download_location_data.py white-sands
```

**New workflow (Option 2 - Using wrapper):**
```bash
cd earth/white_sands
python download_white_sands_data_new.py
```

## Command-Line Options

All command-line options from the old scripts are supported:

| Old Option | New Option | Notes |
|------------|------------|-------|
| `--config PATH` | `--config PATH` | Same |
| `--output-base PATH` | `--output-base PATH` | Same |
| `--stop-after N` | `--stop-after N` | Same |
| `--timeout SECONDS` | `--timeout SECONDS` | Same |
| N/A | `--locations-file PATH` | New: specify custom locations file |

## New Capabilities

The new system adds several capabilities not available in the old system:

### 1. Easy Configuration Generation

Generate custom configurations without manual editing:

```bash
# Generate Ann Arbor config with custom dates
python generate_config.py ann-arbor \
    --start-date 2025-12-01 \
    --end-date 2025-12-03 \
    --output custom_ann_arbor.yaml

# Generate White Sands config with higher resolution
python generate_config.py white-sands \
    --nx2 500 --nx3 400 \
    --output white_sands_highres.yaml
```

### 2. List Available Locations

See all configured locations:

```bash
python generate_config.py --list
```

### 3. Add New Locations Without Code

Add a new location by editing `locations.yaml`:

```yaml
locations:
  my-new-site:
    name: "My Test Site"
    description: "Description here"
    polygon:
      - [lon1, lat1]
      - [lon2, lat2]
      - [lon3, lat3]
      - [lon4, lat4]
    center:
      longitude: -99.5
      latitude: 35.0
    # ... other settings
```

Then use it immediately:
```bash
python download_location_data.py my-new-site
```

## Backward Compatibility

### Old Scripts Still Work

The original location-specific scripts are still present and functional:
- `earth/ann_arbor/download_ann_arbor_data.py`
- `earth/white_sands/download_white_sands_data.py`

You can continue using them if needed. However, they are no longer maintained and users are encouraged to migrate to the unified system.

### Wrapper Scripts

New wrapper scripts provide backward compatibility:
- `earth/ann_arbor/download_ann_arbor_data_new.py`
- `earth/white_sands/download_white_sands_data_new.py`

These wrappers call the unified script with the appropriate location ID.

## Step-by-Step Migration

### For Scripts and Automation

If you have scripts or automation that call the old scripts:

**Option 1: Update to new command (recommended)**
```bash
# Old
python earth/ann_arbor/download_ann_arbor_data.py --timeout 7200

# New
python earth/download_location_data.py ann-arbor --timeout 7200
```

**Option 2: Use wrapper (temporary)**
```bash
# Old
python earth/ann_arbor/download_ann_arbor_data.py --timeout 7200

# Temporary bridge using wrapper
python earth/ann_arbor/download_ann_arbor_data_new.py --timeout 7200
```

### For Custom Configurations

If you have custom configuration files:

1. **Keep using them** - The new system supports custom configs:
   ```bash
   python download_location_data.py ann-arbor --config my_custom.yaml
   ```

2. **Or regenerate them** with the new generator:
   ```bash
   python generate_config.py ann-arbor \
       --start-date YYYY-MM-DD \
       --nx2 300 \
       --output my_custom.yaml
   ```

### For Documentation

Update your documentation to reference:
- `README_UNIFIED_SYSTEM.md` for the new system
- Location-specific READMEs still contain location details
- Both old and new approaches are documented

## Benefits of Migration

Migrating to the unified system provides:

1. **Reduced Maintenance**: Updates happen in one place
2. **Consistency**: Same interface for all locations
3. **Flexibility**: Command-line overrides for any parameter
4. **Extensibility**: Easy to add new locations
5. **Better Error Messages**: More helpful validation and errors

## Common Questions

### Q: Do I need to change my existing YAML files?
**A:** No, existing YAML configuration files work with both old and new scripts.

### Q: Can I still use the old download scripts?
**A:** Yes, the old scripts are still present and functional, but not actively maintained.

### Q: How do I add a new location?
**A:** Edit `earth/locations.yaml`, add your location, then run:
```bash
python generate_config.py your-location-id
python download_location_data.py your-location-id
```

### Q: What if I find a bug in the new system?
**A:** Please report it as a GitHub issue. You can temporarily use the old scripts while we fix it.

### Q: Are the wrapper scripts permanent?
**A:** Wrappers are provided for transition. They may be deprecated in the future after full migration.

### Q: How do I know which script I'm using?
**A:** The new unified script prints the location name in its banner:
```
======================================================================
Ann Arbor Weather Data Pipeline
======================================================================
Location ID: ann-arbor
...
```

## Support

For issues or questions:
- Check `README_UNIFIED_SYSTEM.md` for detailed documentation
- See location-specific READMEs for location details
- Review `locations.yaml` for all configured locations
- Open a GitHub issue for bugs or feature requests

## Timeline

- **Now**: Both old and new systems available
- **Recommended**: Start using new system for new workflows
- **Future**: Old location-specific scripts may be deprecated

## Examples

### Example 1: Simple Migration

**Old:**
```bash
cd earth/ann_arbor
python download_ann_arbor_data.py
```

**New:**
```bash
cd earth
python download_location_data.py ann-arbor
```

### Example 2: Custom Configuration

**Old:**
```bash
cd earth/ann_arbor
# Edit ann_arbor.yaml manually
python download_ann_arbor_data.py --config ann_arbor.yaml
```

**New:**
```bash
cd earth
python generate_config.py ann-arbor \
    --start-date 2025-11-15 \
    --nx2 300 \
    --output custom.yaml
python download_location_data.py ann-arbor --config custom.yaml
```

### Example 3: Automation Script

**Old automation script:**
```bash
#!/bin/bash
cd /path/to/planets/earth/ann_arbor
python download_ann_arbor_data.py --timeout 7200 || exit 1

cd ../white_sands
python download_white_sands_data.py --timeout 7200 || exit 1
```

**New automation script:**
```bash
#!/bin/bash
cd /path/to/planets/earth

python download_location_data.py ann-arbor --timeout 7200 || exit 1
python download_location_data.py white-sands --timeout 7200 || exit 1
```

## Summary

The unified system provides a better, more maintainable approach while preserving backward compatibility. We recommend migrating to the new system for all new workflows, but you can continue using old scripts during the transition period.

For detailed documentation, see `README_UNIFIED_SYSTEM.md`.
