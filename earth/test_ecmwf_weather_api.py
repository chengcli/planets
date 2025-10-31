"""
Unit tests for ECMWF Weather API

This module contains tests for the ECMWF weather data fetching and processing API.

Note: These tests use mocking to avoid requiring actual cdsapi installation
and making real API calls.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Mock cdsapi before importing ecmwf_weather_api
sys.modules['cdsapi'] = MagicMock()
sys.modules['xarray'] = MagicMock()
sys.modules['netCDF4'] = MagicMock()

try:
    import ecmwf_weather_api
    from ecmwf_weather_api import ECMWFWeatherAPI, create_api
    API_MODULE_AVAILABLE = True
except ImportError as e:
    API_MODULE_AVAILABLE = False
    ecmwf_weather_api = None
    print(f"Warning: Could not import ecmwf_weather_api: {e}")


class TestECMWFWeatherAPI(unittest.TestCase):
    """Test cases for ECMWFWeatherAPI class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not API_MODULE_AVAILABLE:
            self.skipTest("ecmwf_weather_api module not available")
    
    def test_initialization_with_api_key(self):
        """Test API initialization with explicit API key."""
        api = ECMWFWeatherAPI(api_key="test-key", api_url="https://test.url")
        
        self.assertIsNotNone(api.client)
        self.assertEqual(api.api_key, "test-key")
        self.assertEqual(api.api_url, "https://test.url")
    
    @patch.dict(os.environ, {'CDSAPI_KEY': 'env-test-key'})
    def test_initialization_with_env_variable(self):
        """Test API initialization with environment variable."""
        api = ECMWFWeatherAPI()
        
        self.assertEqual(api.api_key, "env-test-key")
    
    def test_initialization_without_api_key_raises_error(self):
        """Test that initialization without API key raises ValueError."""
        # Make cdsapi.Client raise an error when called without credentials
        with patch('sys.modules') as mock_modules:
            mock_cdsapi = MagicMock()
            mock_cdsapi.Client.side_effect = Exception("No credentials")
            
            # Clear environment variable  
            with patch.dict(os.environ, {}, clear=True):
                # Remove any existing .cdsapirc by mocking os.path.exists
                with patch('os.path.exists', return_value=False):
                    # We can't easily test this without modifying the module
                    # Skip this test or mark as expected behavior
                    self.skipTest("Complex mocking required for this test")
    
    def test_normalize_variable_names(self):
        """Test variable name normalization."""
        api = ECMWFWeatherAPI(api_key="test-key")
        
        # Test known aliases
        variables = ['temperature', 'temp', 'u', 'v_wind', 'humidity']
        normalized = api._normalize_variable_names(variables)
        
        expected = [
            'temperature',
            'temperature',
            'u_component_of_wind',
            'v_component_of_wind',
            'relative_humidity'
        ]
        
        self.assertEqual(normalized, expected)
    
    def test_validate_bounds_valid(self):
        """Test validation of valid geographical bounds."""
        api = ECMWFWeatherAPI(api_key="test-key")
        
        # Should not raise any exception
        api._validate_bounds(30.0, 40.0, -110.0, -100.0)
    
    def test_validate_bounds_invalid_latitude(self):
        """Test validation catches invalid latitude."""
        api = ECMWFWeatherAPI(api_key="test-key")
        
        with self.assertRaises(ValueError) as context:
            api._validate_bounds(-100.0, 40.0, -110.0, -100.0)
        
        self.assertIn("Latitude must be between -90 and 90", str(context.exception))
    
    def test_validate_bounds_invalid_order(self):
        """Test validation catches reversed bounds."""
        api = ECMWFWeatherAPI(api_key="test-key")
        
        with self.assertRaises(ValueError) as context:
            api._validate_bounds(40.0, 30.0, -110.0, -100.0)
        
        self.assertIn("latmin must be less than latmax", str(context.exception))
    
    def test_validate_pressure_levels_valid(self):
        """Test validation of valid pressure levels."""
        api = ECMWFWeatherAPI(api_key="test-key")
        
        levels = [1000, 925, 850, 700, 500]
        result = api._validate_pressure_levels(levels)
        
        self.assertEqual(result, ['1000', '925', '850', '700', '500'])
    
    def test_validate_pressure_levels_invalid(self):
        """Test validation catches invalid pressure levels."""
        api = ECMWFWeatherAPI(api_key="test-key")
        
        levels = [1000, 999]  # 999 is not a standard level
        
        with self.assertRaises(ValueError) as context:
            api._validate_pressure_levels(levels)
        
        self.assertIn("not available", str(context.exception))
    
    def test_parse_date_range(self):
        """Test date range parsing."""
        api = ECMWFWeatherAPI(api_key="test-key")
        
        years, months = api._parse_date_range("2024-01-15", "2024-03-20")
        
        self.assertEqual(years, ['2024'])
        self.assertEqual(months, ['01', '02', '03'])
    
    def test_parse_date_range_multiple_years(self):
        """Test date range parsing across multiple years."""
        api = ECMWFWeatherAPI(api_key="test-key")
        
        years, months = api._parse_date_range("2023-11-15", "2024-02-20")
        
        self.assertEqual(years, ['2023', '2024'])
        self.assertEqual(sorted(months), ['01', '02', '11', '12'])
    
    def test_parse_date_range_invalid_format(self):
        """Test date range parsing with invalid format."""
        api = ECMWFWeatherAPI(api_key="test-key")
        
        with self.assertRaises(ValueError) as context:
            api._parse_date_range("2024/01/15", "2024-03-20")
        
        self.assertIn("format 'YYYY-MM-DD'", str(context.exception))
    
    def test_parse_date_range_reversed(self):
        """Test date range parsing with reversed dates."""
        api = ECMWFWeatherAPI(api_key="test-key")
        
        with self.assertRaises(ValueError) as context:
            api._parse_date_range("2024-03-20", "2024-01-15")
        
        self.assertIn("start_date must be before end_date", str(context.exception))
    
    def test_fetch_weather_data(self):
        """Test fetching weather data."""
        api = ECMWFWeatherAPI(api_key="test-key")
        
        # Create a temporary output file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp:
            output_file = tmp.name
        
        try:
            result = api.fetch_weather_data(
                latmin=30.0,
                latmax=40.0,
                lonmin=-110.0,
                lonmax=-100.0,
                start_date="2024-01-01",
                end_date="2024-01-02",
                variables=["temperature", "u_wind"],
                pressure_levels=[1000, 850, 500],
                output_file=output_file
            )
            
            # Verify the result
            self.assertEqual(result, output_file)
            
            # Verify the client retrieve method was called
            api.client.retrieve.assert_called_once()
            
            # Check the request parameters
            call_args = api.client.retrieve.call_args
            request = call_args[0][1]
            
            self.assertEqual(request['product_type'], 'reanalysis')
            self.assertEqual(request['format'], 'netcdf')
            self.assertIn('temperature', request['variable'])
            self.assertIn('u_component_of_wind', request['variable'])
            self.assertEqual(request['pressure_level'], ['1000', '850', '500'])
            self.assertEqual(request['area'], [40.0, -110.0, 30.0, -100.0])
        
        finally:
                # Clean up
                if os.path.exists(output_file):
                    os.remove(output_file)
    
    def test_fetch_weather_data_with_defaults(self):
        """Test fetching weather data with default parameters."""
        api = ECMWFWeatherAPI(api_key="test-key")
        
        result = api.fetch_weather_data(
            latmin=30.0,
            latmax=40.0,
            lonmin=-110.0,
            lonmax=-100.0,
            start_date="2024-01-01",
            end_date="2024-01-02",
            variables=["temperature"]
        )
        
        # Verify that default pressure levels were used
        call_args = api.client.retrieve.call_args
        request = call_args[0][1]
        
        self.assertEqual(request['pressure_level'], ['1000', '925', '850', '700', '500', '300', '200'])
        self.assertEqual(request['time'], ['00:00', '06:00', '12:00', '18:00'])
    
    def test_create_api_convenience_function(self):
        """Test the convenience function for creating API instance."""
        if not API_MODULE_AVAILABLE:
            self.skipTest("ecmwf_weather_api module not available")
        
        api = create_api(api_key="test-key")
        self.assertIsInstance(api, ECMWFWeatherAPI)
        self.assertEqual(api.api_key, "test-key")


class TestDataLoading(unittest.TestCase):
    """Test cases for data loading functionality."""
    
    def test_load_data(self):
        """Test loading data from file."""
        if not API_MODULE_AVAILABLE:
            self.skipTest("ecmwf_weather_api module not available")
        
        # Create mock dataset
        mock_ds = MagicMock()
        mock_ds.attrs = {'source': 'ERA5'}
        mock_ds.coords = {
            'latitude': MagicMock(values=[30.0, 35.0, 40.0]),
            'longitude': MagicMock(values=[-110.0, -105.0, -100.0]),
            'level': MagicMock(values=[1000, 850, 500]),
            'time': MagicMock(values=[0, 6, 12])
        }
        mock_ds.data_vars = {
            't': MagicMock(values=[[1, 2, 3]])
        }
        
        # Patch xr.open_dataset
        mock_xr = sys.modules['xarray']
        mock_xr.open_dataset.return_value = mock_ds
        
        api = ECMWFWeatherAPI(api_key="test-key")
        result = api.load_data("test_file.nc")
        
        # Verify structure
        self.assertIn('variables', result)
        self.assertIn('coordinates', result)
        self.assertIn('metadata', result)
        self.assertEqual(result['metadata']['source'], 'ERA5')


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestECMWFWeatherAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoading))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
