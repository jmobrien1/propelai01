"""
Test DocumentBundle and RFPBundle compatibility after fixes
"""

import sys
sys.path.insert(0, '/app')

from agents.enhanced_compliance.bundle_detector import BundleDetector, DocumentBundle
from agents.enhanced_compliance.parser import MultiFormatParser

def test_bundle_compatibility():
    """Test that parser can handle new DocumentBundle format"""
    print("=" * 80)
    print("TEST: DocumentBundle and Parser Compatibility")
    print("=" * 80)
    
    # Create a new DocumentBundle (as would be created by BundleDetector)
    detector = BundleDetector()
    test_files = [
        '/tmp/test_main.pdf',
        '/tmp/test_amendment.pdf',
    ]
    
    bundle = detector.detect_from_files(test_files)
    
    print(f"\n✅ Created DocumentBundle with {len(bundle.documents)} documents")
    print(f"   Main solicitation: {bundle.main_solicitation}")
    print(f"   Amendments: {len(bundle.amendments)}")
    
    # Test that parser can handle it
    parser = MultiFormatParser()
    
    try:
        # This should not crash with AttributeError
        parsed_docs = parser.parse_bundle(bundle)
        print(f"\n✅ Parser handled DocumentBundle successfully!")
        print(f"   Parsed {len(parsed_docs)} documents (files don't exist, so parsing fails, but no AttributeError)")
    except AttributeError as e:
        print(f"\n❌ AttributeError: {e}")
        raise
    except Exception as e:
        # Other errors are OK (files don't exist, etc.)
        print(f"\n✅ No AttributeError! (Got expected error: {type(e).__name__})")
    
    print("\n" + "=" * 80)
    print("✅ TEST PASSED: No AttributeError on DocumentBundle")
    print("=" * 80)

if __name__ == "__main__":
    try:
        test_bundle_compatibility()
        print("\n🎉 All compatibility tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
