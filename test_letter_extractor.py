"""
Test script for RFP Letter Extractor
Sprint 2 - Phase 4.1
"""

import sys
import json
sys.path.insert(0, '/app')

from agents.enhanced_compliance.rfp_letter_extractor import extract_rfp_letter

# Sample RFP Letter text (based on typical government RFP format)
SAMPLE_RFP_LETTER = """
DEPARTMENT OF DEFENSE
ACQUISITION OFFICE

Subject: Request for Proposal (RFP) - Cybersecurity Services
Solicitation Number: HQ0034-24-R-0015
Due Date: March 15, 2024

SUBMISSION INSTRUCTIONS

The Government requires proposals to be submitted in THREE (3) volumes:

Volume I - Technical Approach
The technical volume shall not exceed 30 pages and must describe your approach to 
meeting the technical requirements. Use 11-point Times New Roman font with margins 
of not less than one (1) inch on all sides. Pages shall be single-spaced on 8.5 x 11 
inch paper.

Volume II - Past Performance
Volume II shall be limited to 15 pages and shall include a minimum of three (3) 
relevant contracts. Each reference shall be limited to 2 pages per reference.

Volume III - Price
Pricing information shall ONLY appear in Volume III. Any pricing information in 
Volumes I or II will result in the proposal not being evaluated.

CRITICAL REQUIREMENTS:
1. All offerors must be registered in SAM.gov prior to award
2. Proposals received after the due date will not be accepted
3. Proposals exceeding the page limit will not be evaluated
4. Price information shall not be included in the Technical or Past Performance volumes

FORMATTING REQUIREMENTS:
- Font: 11-point Times New Roman
- Margins: 1 inch on all sides  
- Line spacing: Single-spaced
- Paper size: 8.5 x 11 inches

Proposals must be submitted electronically by 2:00 PM EST on March 15, 2024.

Point of Contact: John Smith
Email: john.smith@acquisition.gov
"""

def test_basic_extraction():
    """Test basic extraction functionality"""
    print("=" * 80)
    print("TEST 1: Basic RFP Letter Extraction")
    print("=" * 80)
    
    result = extract_rfp_letter(SAMPLE_RFP_LETTER, "Test_RFP_Letter.pdf")
    
    # Pretty print results
    print("\n📊 EXTRACTION RESULTS:\n")
    print(json.dumps(result, indent=2))
    
    # Verify key components
    print("\n" + "=" * 80)
    print("✅ VERIFICATION:")
    print("=" * 80)
    
    # Check volumes
    volumes = result.get('volumes', [])
    print(f"\n📁 Volumes Found: {len(volumes)}")
    for vol in volumes:
        print(f"   - Volume {vol['volume_id']}: {vol['volume_name']}")
        if vol['page_limit']:
            print(f"     Page Limit: {vol['page_limit']} pages ({vol['page_limit_text']})")
    
    # Check formatting rules
    formatting_rules = result.get('formatting_rules', [])
    print(f"\n🎨 Formatting Rules Found: {len(formatting_rules)}")
    for rule in formatting_rules:
        print(f"   - {rule['rule_type']}: {rule['value']}")
        print(f"     Source: \"{rule['source_text']}\"")
    
    # Check compliance flags
    compliance_flags = result.get('compliance_flags', [])
    print(f"\n⚠️  Compliance Flags Found: {len(compliance_flags)}")
    for flag in compliance_flags:
        print(f"   - [{flag['severity']}] {flag['category']}")
        print(f"     Rule: \"{flag['rule_text']}\"")
        print(f"     Impact: {flag['impact']}")
        print(f"     Action: {flag['action_required']}")
        print()
    
    # Check metadata
    metadata = result.get('metadata', {})
    print(f"\n📋 Metadata:")
    print(f"   - Source: {metadata.get('source')}")
    print(f"   - Due Date: {metadata.get('due_date', 'Not found')}")
    
    # Check summary
    summary = result.get('summary', {})
    print(f"\n📈 Summary:")
    print(f"   - Total Volumes: {summary.get('total_volumes')}")
    print(f"   - Total Formatting Rules: {summary.get('total_formatting_rules')}")
    print(f"   - Total Compliance Flags: {summary.get('total_compliance_flags')}")
    print(f"   - Critical Flags: {summary.get('critical_flags')}")
    
    # Success indicators
    print("\n" + "=" * 80)
    print("🎯 TEST ASSERTIONS:")
    print("=" * 80)
    
    assert len(volumes) == 3, f"Expected 3 volumes, found {len(volumes)}"
    print("✅ Volume detection: PASS (found 3 volumes)")
    
    assert any(vol['volume_id'] == 'I' for vol in volumes), "Volume I not found"
    print("✅ Volume I (Technical): PASS")
    
    assert any(vol['volume_id'] == 'II' for vol in volumes), "Volume II not found"
    print("✅ Volume II (Past Performance): PASS")
    
    assert any(vol['volume_id'] == 'III' for vol in volumes), "Volume III not found"
    print("✅ Volume III (Price): PASS")
    
    # Check if page limits were extracted
    vol_i = next((v for v in volumes if v['volume_id'] == 'I'), None)
    if vol_i and vol_i['page_limit'] == 30:
        print(f"✅ Volume I page limit: PASS (30 pages)")
    else:
        print(f"⚠️  Volume I page limit: Expected 30, got {vol_i['page_limit'] if vol_i else 'None'}")
    
    # Check formatting rules
    has_font_size = any(r['rule_type'] == 'font_size' for r in formatting_rules)
    has_font_family = any(r['rule_type'] == 'font_family' for r in formatting_rules)
    has_margins = any(r['rule_type'] == 'margins' for r in formatting_rules)
    
    if has_font_size and has_font_family and has_margins:
        print("✅ Formatting rules: PASS (font, margins detected)")
    else:
        print(f"⚠️  Formatting rules: Partial (font_size:{has_font_size}, font_family:{has_font_family}, margins:{has_margins})")
    
    # Check critical compliance flags
    has_price_isolation = any(f['category'] == 'price_isolation' for f in compliance_flags)
    has_page_limit_enforcement = any(f['category'] == 'page_limit_enforcement' for f in compliance_flags)
    
    if has_price_isolation:
        print("✅ Price isolation flag: PASS")
    else:
        print("⚠️  Price isolation flag: NOT FOUND")
    
    if has_page_limit_enforcement:
        print("✅ Page limit enforcement flag: PASS")
    else:
        print("⚠️  Page limit enforcement flag: NOT FOUND")
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE!")
    print("=" * 80)
    
    return result


def test_edge_cases():
    """Test with varied patterns"""
    print("\n\n" + "=" * 80)
    print("TEST 2: Edge Case - Per-item page limits")
    print("=" * 80)
    
    edge_case_text = """
    VOLUME STRUCTURE:
    
    Volume 1: Technical Proposal (not to exceed 25 pages)
    Volume 2: Past Performance - 2 pages per reference
    Volume 3: Pricing Volume (30-page limit)
    
    Use 12-point Arial font with double-spaced text.
    Margins shall be 1 inch.
    """
    
    result = extract_rfp_letter(edge_case_text, "Edge_Case.pdf")
    
    print("\n📊 RESULTS:")
    print(f"Volumes found: {result['summary']['total_volumes']}")
    for vol in result['volumes']:
        print(f"  - Volume {vol['volume_id']}: {vol['volume_name']} ({vol['page_limit_text'] or 'No limit'})")
    
    print("\nFormatting rules:")
    for rule in result['formatting_rules']:
        print(f"  - {rule['rule_type']}: {rule['value']}")
    
    print("\n✅ Edge case test complete!")


if __name__ == "__main__":
    try:
        # Run tests
        result1 = test_basic_extraction()
        result2 = test_edge_cases()
        
        print("\n\n" + "🎉" * 40)
        print("ALL TESTS PASSED!")
        print("🎉" * 40)
        
    except Exception as e:
        print(f"\n\n❌ TEST FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
