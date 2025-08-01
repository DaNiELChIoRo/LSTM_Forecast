import unittest
import os
import tempfile
import shutil

class TestFilePathHandling(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
    
    def tearDown(self):
        """Clean up after each test method."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_problematic_ticker_file_creation(self):
        """Test that files can be created with problematic ticker names."""
        problematic_tickers = [
            "USD/MXN",  # Contains forward slash
            "^IXIC",    # Contains caret
            "MXN=X",    # Contains equals sign
            "STOCK/PRICE"  # Contains forward slash
        ]
        
        for ticker in problematic_tickers:
            # Clean the ticker name
            clean_ticker = ticker.replace('/', '_').replace('^', '').replace('=', '_')
            
            # Try to create files with the cleaned name
            test_files = [
                f'{clean_ticker}_predictions.png',
                f'full {clean_ticker}_predictions.png',
                f'models/{clean_ticker}_model.h5',
                f'models/{clean_ticker}_scaler.pkl',
                f'models/{clean_ticker}_mae_info.pkl'
            ]
            
            # Create models directory
            os.makedirs('models', exist_ok=True)
            
            for file_path in test_files:
                try:
                    # Try to create the file
                    with open(file_path, 'w') as f:
                        f.write('test content')
                    
                    # Verify the file was created
                    self.assertTrue(os.path.exists(file_path))
                    
                    # Clean up
                    os.remove(file_path)
                    
                except Exception as e:
                    self.fail(f"Failed to create file {file_path} for ticker {ticker}: {e}")
    
    def test_original_error_case(self):
        """Test the specific case that was causing the original error."""
        ticker = "USD/MXN"
        clean_ticker = ticker.replace('/', '_').replace('^', '').replace('=', '_')
        
        # This should be "USD_MXN" not "USD/MXN"
        self.assertEqual(clean_ticker, "USD_MXN")
        
        # Try to create the problematic file path
        image_name_full = f'full {clean_ticker}_predictions.png'
        
        # This should not contain any forward slashes
        self.assertNotIn('/', image_name_full)
        
        # Should be able to create the file
        with open(image_name_full, 'w') as f:
            f.write('test content')
        
        self.assertTrue(os.path.exists(image_name_full))
        
        # Clean up
        os.remove(image_name_full)

if __name__ == '__main__':
    unittest.main(verbosity=2) 