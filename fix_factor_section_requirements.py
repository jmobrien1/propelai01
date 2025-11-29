#!/usr/bin/env python3
"""
PropelAI: Fix Factor Section Requirements - Targeted Patch

DIAGNOSIS:
The annotated outline shows Factor sections (SEC-F1 through SEC-F6) being
generated but with EMPTY requirements blocks. The issue is that when semantic
matching finds no results (score threshold too high or keywords not matching),
the output block is skipped entirely due to the conditional check.

This patch:
1. Adds an else block to always output requirements (with fallback)
2. Lowers the minimum threshold slightly  
3. Adds debug logging
4. Ensures EVERY factor section shows relevant requirements

Run from propelai directory:
    python3 fix_factor_section_requirements.py
"""

import os
import re

def apply_fix():
    filepath = "agents/enhanced_compliance/annotated_outline_exporter.js"
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Backup
    backup_path = filepath + '.backup_fix'
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"✅ Created backup: {backup_path}")
    
    # ===========================================================
    # FIX: Add else block to always output requirements
    # ===========================================================
    
    # Find the conditional block that only outputs if pwsReqs.length > 0
    old_block = '''    if (pwsReqs.length > 0) {
        children.push(createAnnotationBlock(
            "SECTION C/PWS - TECHNICAL REQUIREMENTS",
            pwsReqs.map(r => `[${r.req_id || 'REQ'}] ${r.text || r.full_text || ''}`),
            COLORS.SECTION_C,
            ANNOTATION_SHADING.C
        ));
    }

    // === WIN THEMES PLACEHOLDER (GREEN) ==='''
    
    new_block = '''    // Always output requirements - semantic matches or fallback
    if (pwsReqs.length > 0) {
        console.log(`[OUTLINE] Factor "${sectionName}": Found ${pwsReqs.length} semantic matches`);
        children.push(createAnnotationBlock(
            "SECTION C/PWS - TECHNICAL REQUIREMENTS",
            pwsReqs.map(r => {
                const reqId = r.req_id || r.id || 'REQ';
                const text = r.text || r.full_text || '[Requirement text]';
                return `[${reqId}] ${text.substring(0, 300)}${text.length > 300 ? '...' : ''}`;
            }),
            COLORS.SECTION_C,
            ANNOTATION_SHADING.C
        ));
    } else if (requirements && requirements.length > 0) {
        // Fallback: Show first 5-8 requirements if semantic matching found nothing
        console.log(`[OUTLINE] Factor "${sectionName}": No semantic matches - using fallback (${requirements.length} total requirements)`);
        const fallbackReqs = requirements.slice(0, 8);
        children.push(createAnnotationBlock(
            "SECTION C/PWS - TECHNICAL REQUIREMENTS (Review for relevance)",
            fallbackReqs.map(r => {
                const reqId = r.req_id || r.id || 'REQ';
                const text = r.text || r.full_text || '[Requirement text]';
                return `[${reqId}] ${text.substring(0, 250)}${text.length > 250 ? '...' : ''}`;
            }),
            COLORS.SECTION_C,
            ANNOTATION_SHADING.C
        ));
    } else {
        // No requirements available at all - show placeholder
        console.log(`[OUTLINE] Factor "${sectionName}": WARNING - No requirements to display`);
        children.push(createAnnotationBlock(
            "SECTION C/PWS - TECHNICAL REQUIREMENTS",
            [
                "[No requirements extracted from RFP - manually review Section C/PWS]",
                "[Map specific technical requirements relevant to this evaluation factor]",
                "[Ensure all SHALL/MUST requirements are addressed]"
            ],
            COLORS.SECTION_C,
            ANNOTATION_SHADING.C,
            true  // isPlaceholder
        ));
    }

    // === WIN THEMES PLACEHOLDER (GREEN) ==='''
    
    if old_block in content:
        content = content.replace(old_block, new_block)
        print("✅ Added fallback logic to always output requirements")
    else:
        print("⚠️ Could not find exact pattern - checking for similar structure...")
        
        # Try to find just the if block
        pattern = r'if \(pwsReqs\.length > 0\) \{\s*children\.push\(createAnnotationBlock\('
        if re.search(pattern, content):
            print("   Found conditional block but structure differs")
            print("   Manual review recommended")
        else:
            print("   Pattern not found - file may have been modified")
        
        return False
    
    # ===========================================================
    # ADDITIONAL FIX: Lower the score threshold slightly
    # ===========================================================
    
    # Change threshold from > 5 to >= 3
    old_threshold = '.filter(r => r._score > 5)  // Minimum relevance threshold'
    new_threshold = '.filter(r => r._score >= 3)  // Minimum relevance threshold (lowered to capture more)'
    
    if old_threshold in content:
        content = content.replace(old_threshold, new_threshold)
        print("✅ Lowered relevance threshold from >5 to >=3")
    
    # ===========================================================
    # ADDITIONAL FIX: Add debug logging at function start
    # ===========================================================
    
    old_func_start = '''function buildSectionOutline(section, secIndex, volume, requirements, data) {
    const children = [];'''
    
    new_func_start = '''function buildSectionOutline(section, secIndex, volume, requirements, data) {
    const children = [];
    
    // Debug logging
    const sectionDebugInfo = section.name || section.title || section.id || 'Unknown';
    console.log(`[OUTLINE] Building section: ${sectionDebugInfo}`);
    console.log(`[OUTLINE]   Requirements passed: ${requirements ? requirements.length : 'NONE'}`);'''