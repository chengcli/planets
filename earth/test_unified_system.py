#!/usr/bin/env python3
"""
Tests for the unified location configuration system.

These tests validate:
- Location table loading
- Configuration generation
- Location ID validation
- Command-line parsing
"""

import unittest
import sys
import tempfile
import shutil
from pathlib import Path

# Add earth directory to path
EARTH_DIR = Path(__file__).parent
sys.path.insert(0, str(EARTH_DIR))

# Import our modules
import generate_config


class TestLocationTable(unittest.TestCase):
    """Test location table loading and validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.locations_file = EARTH_DIR / "locations.csv"
        self.template_file = EARTH_DIR / "config_template.yaml"
    
    def test_locations_file_exists(self):
        """Test that locations.csv exists."""
        csv_file = EARTH_DIR / "locations.csv"
        self.assertTrue(csv_file.exists())
    
    def test_template_file_exists(self):
        """Test that config_template.yaml exists."""
        self.assertTrue(self.template_file.exists())
    
    def test_load_locations(self):
        """Test loading locations from CSV."""
        locations = generate_config.load_locations(self.locations_file)
        self.assertIsInstance(locations, dict)
        self.assertIn('ann-arbor', locations)
        self.assertIn('white-sands', locations)
    
    def test_location_structure(self):
        """Test that locations have required fields."""
        locations = generate_config.load_locations(self.locations_file)
        
        for loc_id, loc_data in locations.items():
            # Check required fields (simplified structure)
            self.assertIn('name', loc_data)
            self.assertIn('polygon', loc_data)
            
            # Check polygon has vertices
            self.assertIsInstance(loc_data['polygon'], list)
            self.assertGreater(len(loc_data['polygon']), 0)
    
    def test_calculate_center(self):
        """Test center calculation from polygon."""
        locations = generate_config.load_locations(self.locations_file)
        
        for loc_id, loc_data in locations.items():
            center = generate_config.calculate_center(loc_data['polygon'])
            self.assertIn('latitude', center)
            self.assertIn('longitude', center)
            self.assertIsInstance(center['latitude'], float)
            self.assertIsInstance(center['longitude'], float)


class TestLocationIDValidation(unittest.TestCase):
    """Test location ID validation rules."""
    
    def test_valid_ids(self):
        """Test that valid location IDs pass validation."""
        valid_ids = [
            'ann-arbor',
            'white-sands',
            'test_site',
            'location123',
            'site-A_1',
        ]
        
        for loc_id in valid_ids:
            try:
                generate_config.validate_location_id(loc_id)
            except ValueError as e:
                self.fail(f"Valid ID '{loc_id}' was rejected: {e}")
    
    def test_invalid_ids(self):
        """Test that invalid location IDs are rejected."""
        invalid_ids = [
            'Ann Arbor',      # space
            'white.sands',    # period
            'site@location',  # @
            'test/site',      # slash
            'location!',      # exclamation
            'a b c',          # spaces
        ]
        
        for loc_id in invalid_ids:
            with self.assertRaises(ValueError):
                generate_config.validate_location_id(loc_id)


class TestDateValidation(unittest.TestCase):
    """Test date format validation."""
    
    def test_valid_dates(self):
        """Test that valid dates pass validation."""
        valid_dates = [
            '2025-01-01',
            '2024-12-31',
            '2025-10-15',
        ]
        
        for date in valid_dates:
            try:
                generate_config.validate_date_format(date)
            except ValueError as e:
                self.fail(f"Valid date '{date}' was rejected: {e}")
    
    def test_invalid_dates(self):
        """Test that invalid dates are rejected."""
        invalid_dates = [
            '2025-1-1',       # missing leading zeros
            '25-01-01',       # 2-digit year
            '2025/01/01',     # wrong separator
            '01-01-2025',     # wrong order
            'not-a-date',     # not a date
        ]
        
        for date in invalid_dates:
            with self.assertRaises(ValueError):
                generate_config.validate_date_format(date)


class TestConfigGeneration(unittest.TestCase):
    """Test configuration file generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.locations_file = EARTH_DIR / "locations.csv"
        self.template_file = EARTH_DIR / "config_template.yaml"
        self.locations = generate_config.load_locations(self.locations_file)
        self.template = generate_config.load_template(self.template_file)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_ann_arbor_config(self):
        """Test generating Ann Arbor configuration."""
        # Mock args with required parameters
        class Args:
            start_date = '2025-11-01'
            end_date = '2025-11-01'
            nx1 = 150
            nx2 = 200
            nx3 = 200
            nghost = 3
            x1_max = 15000.0
            x2_extent = 125000.0
            x3_extent = 125000.0
            tlim = 43200
        
        args = Args()
        config = generate_config.generate_config(
            'ann-arbor', self.locations, self.template, args
        )
        
        # Check that placeholders were replaced
        self.assertNotIn('{location_name}', config)
        self.assertNotIn('{center_latitude}', config)
        self.assertIn('Ann Arbor', config)
    
    def test_generate_white_sands_config(self):
        """Test generating White Sands configuration."""
        class Args:
            start_date = '2025-10-01'
            end_date = '2025-10-02'
            nx1 = 150
            nx2 = 400
            nx3 = 300
            nghost = 3
            x1_max = 15000.0
            x2_extent = 222640.0
            x3_extent = 139800.0
            tlim = 172800
        
        args = Args()
        config = generate_config.generate_config(
            'white-sands', self.locations, self.template, args
        )
        
        # Check that placeholders were replaced
        self.assertNotIn('{location_name}', config)
        self.assertNotIn('{center_latitude}', config)
        self.assertIn('White Sands', config)
    
    def test_generate_with_overrides(self):
        """Test generating config with command-line overrides."""
        class Args:
            start_date = '2025-12-01'
            end_date = '2025-12-03'
            nx1 = 200
            nx2 = 300
            nx3 = 300
            nghost = 3
            x1_max = 15000.0
            x2_extent = 125000.0
            x3_extent = 125000.0
            tlim = 172800
        
        args = Args()
        config = generate_config.generate_config(
            'ann-arbor', self.locations, self.template, args
        )
        
        # Check that overrides were applied
        self.assertIn('2025-12-01', config)
        self.assertIn('2025-12-03', config)
        self.assertIn('nx1: 200', config)
        self.assertIn('nx2: 300', config)
        self.assertIn('nx3: 300', config)
        self.assertIn('tlim: 172800', config)
    
    def test_invalid_location(self):
        """Test that invalid location raises error."""
        class Args:
            start_date = '2025-11-01'
            end_date = '2025-11-01'
            nx1 = 150
            nx2 = 200
            nx3 = 200
            nghost = 3
            x1_max = 15000.0
            x2_extent = 125000.0
            x3_extent = 125000.0
            tlim = 43200
        
        args = Args()
        with self.assertRaises(ValueError):
            generate_config.generate_config(
                'invalid-location', self.locations, self.template, args
            )


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
