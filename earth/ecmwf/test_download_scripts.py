"""
Unit tests for ERA5 download convenience scripts.

This module contains tests for the download_era5_wind_temp.py and
download_era5_density_vars.py convenience scripts.
"""

import unittest
import sys
import os
from io import StringIO
from unittest.mock import patch, MagicMock

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Mock modules before imports
sys.modules['cdsapi'] = MagicMock()
sys.modules['xarray'] = MagicMock()
sys.modules['netCDF4'] = MagicMock()

try:
    import download_era5_wind_temp
    import download_era5_density_vars
    SCRIPTS_AVAILABLE = True
except ImportError as e:
    SCRIPTS_AVAILABLE = False
    print(f"Warning: Could not import download scripts: {e}")


class TestDownloadScripts(unittest.TestCase):
    """Test cases for ERA5 download convenience scripts."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not SCRIPTS_AVAILABLE:
            self.skipTest("Download scripts not available")
    
    def test_wind_temp_validate_date_format(self):
        """Test date format validation in wind_temp script."""
        self.assertTrue(download_era5_wind_temp.validate_date_format('2024-01-01'))
        self.assertTrue(download_era5_wind_temp.validate_date_format('2023-12-31'))
        self.assertFalse(download_era5_wind_temp.validate_date_format('2024-13-01'))
        self.assertFalse(download_era5_wind_temp.validate_date_format('2024/01/01'))
        self.assertFalse(download_era5_wind_temp.validate_date_format('not-a-date'))
    
    def test_density_vars_validate_date_format(self):
        """Test date format validation in density_vars script."""
        self.assertTrue(download_era5_density_vars.validate_date_format('2024-01-01'))
        self.assertTrue(download_era5_density_vars.validate_date_format('2023-12-31'))
        self.assertFalse(download_era5_density_vars.validate_date_format('2024-13-01'))
        self.assertFalse(download_era5_density_vars.validate_date_format('2024/01/01'))
        self.assertFalse(download_era5_density_vars.validate_date_format('not-a-date'))
    
    @patch('sys.argv', ['download_era5_wind_temp.py', '--help'])
    def test_wind_temp_help_argument(self):
        """Test that wind_temp script can display help."""
        with self.assertRaises(SystemExit) as cm:
            download_era5_wind_temp.parse_arguments()
        # Help exits with code 0
        self.assertEqual(cm.exception.code, 0)
    
    @patch('sys.argv', ['download_era5_density_vars.py', '--help'])
    def test_density_vars_help_argument(self):
        """Test that density_vars script can display help."""
        with self.assertRaises(SystemExit) as cm:
            download_era5_density_vars.parse_arguments()
        # Help exits with code 0
        self.assertEqual(cm.exception.code, 0)
    
    @patch('sys.argv', [
        'download_era5_wind_temp.py',
        '--latmin', '30.0',
        '--latmax', '40.0',
        '--lonmin', '-110.0',
        '--lonmax', '-100.0',
        '--start-date', '2024-01-01',
        '--end-date', '2024-01-02',
        '--output', 'test.nc'
    ])
    def test_wind_temp_parse_valid_arguments(self):
        """Test parsing valid arguments for wind_temp script."""
        args = download_era5_wind_temp.parse_arguments()
        
        self.assertEqual(args.latmin, 30.0)
        self.assertEqual(args.latmax, 40.0)
        self.assertEqual(args.lonmin, -110.0)
        self.assertEqual(args.lonmax, -100.0)
        self.assertEqual(args.start_date, '2024-01-01')
        self.assertEqual(args.end_date, '2024-01-02')
        self.assertEqual(args.jobs, 4)  # default value
        self.assertEqual(args.output, 'test.nc')
        self.assertIsNone(args.times)
    
    @patch('sys.argv', [
        'download_era5_density_vars.py',
        '--latmin', '32.0',
        '--latmax', '33.5',
        '--lonmin', '-106.8',
        '--lonmax', '-105.8',
        '--start-date', '2024-01-01',
        '--end-date', '2024-01-02',
        '--output', 'density.nc',
        '--times', '00:00', '12:00'
    ])
    def test_density_vars_parse_arguments_with_times(self):
        """Test parsing arguments with custom times for density_vars script."""
        args = download_era5_density_vars.parse_arguments()
        
        self.assertEqual(args.latmin, 32.0)
        self.assertEqual(args.latmax, 33.5)
        self.assertEqual(args.lonmin, -106.8)
        self.assertEqual(args.lonmax, -105.8)
        self.assertEqual(args.start_date, '2024-01-01')
        self.assertEqual(args.end_date, '2024-01-02')
        self.assertEqual(args.output, 'density.nc')
        self.assertEqual(args.times, ['00:00', '12:00'])
        self.assertEqual(args.jobs, 4)  # default value
    
    @patch('sys.argv', [
        'download_era5_wind_temp.py',
        '--latmin', '30.0',
        '--latmax', '40.0',
        '--lonmin', '-110.0',
        '--lonmax', '-100.0',
        '--start-date', '2024-01-01',
        '--end-date', '2024-01-02',
        '--output', 'test.nc',
        '--jobs', '8'
    ])
    def test_wind_temp_parse_jobs_argument(self):
        """Test parsing custom jobs argument for wind_temp script."""
        args = download_era5_wind_temp.parse_arguments()
        self.assertEqual(args.jobs, 8)
    
    @patch('sys.argv', [
        'download_era5_density_vars.py',
        '--latmin', '30.0',
        '--latmax', '40.0',
        '--lonmin', '-110.0',
        '--lonmax', '-100.0',
        '--start-date', '2024-01-01',
        '--end-date', '2024-01-02',
        '--output', 'test.nc',
        '--jobs', '10'
    ])
    def test_density_vars_parse_jobs_argument(self):
        """Test parsing custom jobs argument for density_vars script."""
        args = download_era5_density_vars.parse_arguments()
        self.assertEqual(args.jobs, 10)
    
    @patch('sys.argv', [
        'download_era5_wind_temp.py',
        '--latmin', '30.0',
        '--latmax', '40.0',
        '--lonmin', '-110.0',
        '--lonmax', '-100.0',
        '--start-date', '2024-13-01',  # Invalid date
        '--end-date', '2024-01-02',
        '--output', 'test.nc'
    ])
    @patch('download_era5_wind_temp.ECMWFWeatherAPI')
    def test_wind_temp_invalid_date_exits(self, mock_api):
        """Test that wind_temp script exits on invalid date."""
        with self.assertRaises(SystemExit) as cm:
            download_era5_wind_temp.main()
        # Should exit with error code 1
        self.assertEqual(cm.exception.code, 1)
    
    @patch('sys.argv', [
        'download_era5_density_vars.py',
        '--latmin', '30.0',
        '--latmax', '40.0',
        '--lonmin', '-110.0',
        '--lonmax', '-100.0',
        '--start-date', '2024-01-01',
        '--end-date', '2024/01/02',  # Invalid date format
        '--output', 'test.nc'
    ])
    @patch('download_era5_density_vars.ECMWFWeatherAPI')
    def test_density_vars_invalid_date_format_exits(self, mock_api):
        """Test that density_vars script exits on invalid date format."""
        with self.assertRaises(SystemExit) as cm:
            download_era5_density_vars.main()
        # Should exit with error code 1
        self.assertEqual(cm.exception.code, 1)
    
    def test_wind_temp_variables_are_correct(self):
        """Test that wind_temp script uses correct variable names."""
        # These should match ECMWF ERA5 variable names
        expected_variables = [
            'temperature',
            'u_component_of_wind',
            'v_component_of_wind'
        ]
        # Check that the script file contains these variables
        script_path = os.path.join(os.path.dirname(__file__), 'download_era5_wind_temp.py')
        with open(script_path, 'r') as f:
            content = f.read()
            for var in expected_variables:
                self.assertIn(var, content)
    
    def test_density_vars_variables_are_correct(self):
        """Test that density_vars script uses correct variable names."""
        # These should match ECMWF ERA5 variable names
        expected_variables = [
            'specific_cloud_ice_water_content',
            'specific_humidity',
            'specific_snow_water_content',
            'specific_cloud_liquid_water_content',
            'specific_rain_water_content'
        ]
        # Check that the script file contains these variables
        script_path = os.path.join(os.path.dirname(__file__), 'download_era5_density_vars.py')
        with open(script_path, 'r') as f:
            content = f.read()
            for var in expected_variables:
                self.assertIn(var, content)
    
    def test_both_scripts_use_all_pressure_levels(self):
        """Test that both scripts use all 37 standard ERA5 pressure levels."""
        expected_levels = [
            1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200,
            225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
            775, 800, 825, 850, 875, 900, 925, 950, 975, 1000
        ]
        
        # Check wind_temp script
        script_path = os.path.join(os.path.dirname(__file__), 'download_era5_wind_temp.py')
        with open(script_path, 'r') as f:
            content = f.read()
            for level in expected_levels:
                self.assertIn(str(level), content)
        
        # Check density_vars script
        script_path = os.path.join(os.path.dirname(__file__), 'download_era5_density_vars.py')
        with open(script_path, 'r') as f:
            content = f.read()
            for level in expected_levels:
                self.assertIn(str(level), content)


if __name__ == '__main__':
    unittest.main()
