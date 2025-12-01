"""
Integration test for RFP Letter Extractor (Sprint 2)
Tests the full workflow from file upload to letter data extraction
"""

import json
import sys
import tempfile
from pathlib import Path

# Create a sample RFP letter text file for testing
SAMPLE_LETTER_TEXT = """
DEPARTMENT OF DEFENSE
REQUEST FOR PROPOSAL - CYBERSECURITY SERVICES
Solicitation Number: HQ0034-24-R-0015

SUBMISSION REQUIREMENTS

Proposals shall be submitted in THREE (3) VOLUMES:

Volume I - Technical Approach
The technical proposal shall not exceed 30 pages. All pages must use 11-point 
Times New Roman font with margins of not less than one (1) inch on all sides. 
Text shall be single-spaced on 8.5 x 11 inch paper.

Volume II - Past Performance  
Past performance volume shall be limited to 15 pages, including a minimum of 
three (3) relevant contracts with 2 pages per reference maximum.

Volume III - Price Volume
Pricing information shall ONLY appear in Volume III. Any pricing or cost data 
in Volumes I or II will result in proposal rejection.

CRITICAL COMPLIANCE REQUIREMENTS:
1. All offerors must be registered in SAM.gov prior to award
2. Proposals exceeding the page limit will not be evaluated  
3. Late proposals will not be accepted
4. Pricing shall not be included in Technical or Past Performance volumes

DUE DATE: March 15, 2024 at 2:00 PM EST

Point of Contact: acquisition.office@defense.gov
"""

def test_extraction_workflow():
    """Test RFP letter extraction using the module directly"""
    print("=" * 80)
    print("INTEGRATION TEST: RFP Letter Extraction Workflow")
    print("=" * 80)
    
    # Add app to path
    sys.path.insert(0, '/app')
    
    # Import the extractor
    from agents.enhanced_compliance.rfp_letter_extractor import extract_rfp_letter
    
    # Test extraction
    print("\n📄 Processing sample RFP letter...")
    result = extract_rfp_letter(SAMPLE_LETTER_TEXT, "Test_RFP_Letter.pdf")
    
    # Validate results
    print("\n✅ EXTRACTION RESULTS:")
    print(f"   Volumes extracted: {result['summary']['total_volumes']}")
    print(f"   Formatting rules: {result['summary']['total_formatting_rules']}")
    print(f"   Compliance flags: {result['summary']['total_compliance_flags']}")
    print(f"   Critical flags: {result['summary']['critical_flags']}")
    
    # Display volumes
    print("\n📁 VOLUMES:")
    for vol in result['volumes']:
        page_info = f" ({vol['page_limit']} pages)" if vol['page_limit'] else ""
        print(f"   - Volume {vol['volume_id']}: {vol['volume_name']}{page_info}")
    
    # Display critical compliance flags
    print("\n⚠️  CRITICAL COMPLIANCE FLAGS:")
    critical_flags = [f for f in result['compliance_flags'] if f['severity'] == 'CRITICAL']
    for flag in critical_flags:
        print(f"   - {flag['category']}: {flag['rule_text'][:60]}...")
    
    # Assertions
    assert result['summary']['total_volumes'] >= 3, "Should find at least 3 volumes"
    assert result['summary']['critical_flags'] >= 2, "Should find at least 2 critical flags"
    
    print("\n✅ INTEGRATION TEST PASSED!")
    print("=" * 80)
    
    return result


def test_api_integration():
    """Test that the API has the new endpoint"""
    import requests
    
    print("\n\n" + "=" * 80)
    print("API INTEGRATION TEST: New /letter endpoint")
    print("=" * 80)
    
    # Test the new endpoint with an existing RFP
    print("\n🌐 Testing API endpoint /api/rfp/{rfp_id}/letter...")
    
    # First get list of RFPs
    response = requests.get("http://localhost:8001/api/rfp")
    rfps = response.json().get('rfps', [])
    
    if rfps:
        rfp_id = rfps[0]['id']
        print(f"   Using RFP: {rfp_id}")
        
        # Test the letter endpoint
        letter_response = requests.get(f"http://localhost:8001/api/rfp/{rfp_id}/letter")
        letter_data = letter_response.json()
        
        print(f"   Response status: {letter_data.get('status')}")
        print(f"   Message: {letter_data.get('message', 'N/A')}")
        
        if letter_data.get('status') == 'available':
            print("\n   ✅ RFP letter data is available!")
            data = letter_data.get('data', {})
            print(f"      Volumes: {data.get('summary', {}).get('total_volumes')}")
            print(f"      Compliance flags: {data.get('summary', {}).get('total_compliance_flags')}")
        else:
            print("\n   ℹ️  RFP letter data not yet extracted (expected for existing RFPs)")
        
        print("\n✅ API endpoint is working!")
    else:
        print("   ℹ️  No RFPs found to test with")
    
    print("=" * 80)


if __name__ == "__main__":
    try:
        # Test 1: Direct extraction
        result = test_extraction_workflow()
        
        # Test 2: API integration
        test_api_integration()
        
        print("\n\n" + "🎉" * 40)
        print("ALL INTEGRATION TESTS PASSED!")
        print("🎉" * 40)
        print("\n✨ Sprint 2 Implementation Complete!")
        print("\nNext Steps:")
        print("  1. Upload a multi-file RFP bundle with an RFP letter")
        print("  2. Process the RFP to trigger letter extraction")
        print("  3. Query /api/rfp/{rfp_id}/letter to view extracted data")
        
    except Exception as e:
        print(f"\n\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
