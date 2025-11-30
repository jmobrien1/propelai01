#!/usr/bin/env python3
"""
Test error handling for Company Library v4.0
"""

import tempfile
from agents.enhanced_compliance.company_library import CompanyLibrary

def test_error_handling():
    """Test error handling scenarios"""
    print("=== Testing Error Handling ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        library = CompanyLibrary(temp_dir)
        
        # Test 1: Non-existent file
        result1 = library.add_document("/nonexistent/file.txt")
        print(f"Non-existent file: {result1['status']} - {result1['message']}")
        assert result1['status'] == 'error'
        assert 'File not found' in result1['message']
        
        # Test 2: Empty file path
        result2 = library.add_document("")
        print(f"Empty path: {result2['status']} - {result2['message']}")
        assert result2['status'] == 'error'
        
        print("=== Error handling tests passed! ===")

if __name__ == "__main__":
    test_error_handling()