#!/usr/bin/env python3
"""
Test script for Company Library v4.0 enhancements
"""

import tempfile
import os
from pathlib import Path
from agents.enhanced_compliance.company_library import CompanyLibrary, DocumentType

def test_duplicate_detection():
    """Test duplicate detection functionality"""
    print("=== Testing Company Library v4.0 ===")
    
    # Create temporary test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create library
        library = CompanyLibrary(temp_dir)
        
        # Create test document
        test_file = Path(temp_dir) / "test_capabilities.txt"
        test_content = """# Acme Corp Capabilities
        
## Executive Summary
Acme Corp provides innovative solutions for government clients.

## Core Capabilities
- **Cloud Migration**: Moving legacy systems to modern cloud platforms
- **Data Analytics**: Advanced analytics and reporting solutions
"""
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        print(f"Created test file: {test_file}")
        
        # Test 1: Add document successfully
        result1 = library.add_document(str(test_file), DocumentType.CAPABILITIES, "Core Services")
        print(f"First add result: {result1['status']} - {result1['message']}")
        assert result1['status'] == 'success'
        
        # Test 2: Try to add same document (should detect duplicate)
        result2 = library.add_document(str(test_file), DocumentType.CAPABILITIES, "Core Services")
        print(f"Duplicate add result: {result2['status']} - {result2['message']}")
        assert result2['status'] == 'duplicate'
        
        # Test 3: Create file with same name but different content
        test_file2 = Path(temp_dir) / "test_capabilities_different.txt"
        test_content2 = """# Acme Corp Capabilities v2
        
## Executive Summary
Acme Corp provides cutting-edge solutions for enterprise clients.

## Core Capabilities
- **AI/ML Solutions**: Machine learning and artificial intelligence
- **Cybersecurity**: Advanced security and compliance solutions
"""
        
        with open(test_file2, 'w') as f:
            f.write(test_content2)
        
        result3 = library.add_document(str(test_file2), DocumentType.CAPABILITIES, "Advanced Services")
        print(f"Different content result: {result3['status']} - {result3['message']}")
        assert result3['status'] == 'success'
        
        # Test 4: Test filename collision (copy first file with same name to library dir)
        collision_file = Path(temp_dir) / "collision_test.txt"
        with open(collision_file, 'w') as f:
            f.write("Different content but same name")
        
        # First, add a file to create the name in library
        result4a = library.add_document(str(collision_file), DocumentType.OTHER, "Test")
        print(f"First collision file: {result4a['status']} - {result4a['message']}")
        
        # Now create another file with same name but different content
        collision_file2 = Path(temp_dir) / "collision_test2.txt"
        with open(collision_file2, 'w') as f:
            f.write("Another different content")
        
        # Rename it to same name as first file
        collision_file_same_name = Path(temp_dir) / "collision_test.txt"
        collision_file2.rename(collision_file_same_name)
        
        result4b = library.add_document(str(collision_file_same_name), DocumentType.OTHER, "Test2")
        print(f"Collision handling: {result4b['status']} - {result4b['message']}")
        
        # Test 5: List documents to verify all were added correctly
        docs = library.list_documents()
        print(f"\nLibrary now contains {len(docs)} documents:")
        for doc in docs:
            print(f"  - {doc['filename']} ({doc['type']}) - {doc['title']}")
        
        print("\n=== All tests passed! ===")

if __name__ == "__main__":
    test_duplicate_detection()