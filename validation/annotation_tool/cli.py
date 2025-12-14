#!/usr/bin/env python3
"""
PropelAI Ground Truth Annotation CLI

Command-line tool for creating and managing ground truth datasets.

Usage:
    python -m validation.annotation_tool.cli new NIH_75N96025R00004 75N96025R00004 --agency NIH
    python -m validation.annotation_tool.cli add NIH_75N96025R00004 --interactive
    python -m validation.annotation_tool.cli list NIH_75N96025R00004
    python -m validation.annotation_tool.cli stats NIH_75N96025R00004
    python -m validation.annotation_tool.cli export NIH_75N96025R00004 --format json
"""

import argparse
import sys
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validation.schemas import (
    GroundTruthRFP,
    GroundTruthRequirement,
    AnnotationStatus,
    BindingLevel,
    RequirementCategory,
)


# ============== Configuration ==============

GROUND_TRUTH_DIR = Path(__file__).parent.parent / "ground_truth" / "rfps"


def ensure_directory(rfp_id: str) -> Path:
    """Ensure the RFP directory exists"""
    rfp_dir = GROUND_TRUTH_DIR / rfp_id
    rfp_dir.mkdir(parents=True, exist_ok=True)
    (rfp_dir / "documents").mkdir(exist_ok=True)
    return rfp_dir


def get_ground_truth_path(rfp_id: str) -> Path:
    """Get the path to the ground truth file"""
    return GROUND_TRUTH_DIR / rfp_id / "ground_truth.json"


def load_or_create_rfp(rfp_id: str) -> Optional[GroundTruthRFP]:
    """Load existing RFP or return None if not found"""
    path = get_ground_truth_path(rfp_id)
    if path.exists():
        return GroundTruthRFP.load(str(path))
    return None


# ============== Commands ==============

def cmd_new(args):
    """Create a new ground truth RFP"""
    rfp_dir = ensure_directory(args.rfp_id)

    # Check if already exists
    gt_path = get_ground_truth_path(args.rfp_id)
    if gt_path.exists():
        print(f"Error: Ground truth already exists for {args.rfp_id}")
        print(f"Use 'edit' to modify or delete the directory first")
        return 1

    # Create new RFP
    rfp = GroundTruthRFP(
        rfp_id=args.rfp_id,
        solicitation_number=args.solicitation_number,
        agency=args.agency or "OTHER",
        rfp_type=args.type or "Full-and-Open",
        document_format=args.format or "UCF_STANDARD",
        title=args.title or "",
        primary_annotator=args.annotator or "",
        annotation_start_date=datetime.now().isoformat(),
    )

    rfp.save(str(gt_path))

    print(f"Created new ground truth RFP: {args.rfp_id}")
    print(f"  Location: {gt_path}")
    print(f"  Agency: {rfp.agency}")
    print(f"  Format: {rfp.document_format}")
    print(f"\nNext steps:")
    print(f"  1. Add RFP documents to: {rfp_dir / 'documents'}")
    print(f"  2. Add requirements: python -m validation.annotation_tool.cli add {args.rfp_id} --interactive")

    return 0


def cmd_add(args):
    """Add a requirement to the ground truth"""
    rfp = load_or_create_rfp(args.rfp_id)
    if not rfp:
        print(f"Error: RFP {args.rfp_id} not found. Create it first with 'new' command.")
        return 1

    if args.interactive:
        return cmd_add_interactive(args.rfp_id, rfp)

    # Add single requirement from command line
    req_id = f"GT-{args.rfp_id[:10]}-{len(rfp.requirements) + 1:03d}"

    req = GroundTruthRequirement(
        gt_id=req_id,
        rfp_id=args.rfp_id,
        text=args.text,
        rfp_section=args.section or "",
        binding_level=args.binding or "Mandatory",
        category=args.category or "TECHNICAL",
        page_number=args.page or 0,
        annotator_id=args.annotator or "",
        annotation_timestamp=datetime.now().isoformat(),
        annotation_status="draft",
    )

    rfp.requirements.append(req)
    rfp.save(str(get_ground_truth_path(args.rfp_id)))

    print(f"Added requirement {req_id}")
    print(f"  Text: {req.text[:80]}...")
    print(f"  Section: {req.rfp_section}")
    print(f"  Binding: {req.binding_level}")

    return 0


def cmd_add_interactive(rfp_id: str, rfp: GroundTruthRFP) -> int:
    """Interactive mode for adding requirements"""
    print(f"\nInteractive annotation mode for {rfp_id}")
    print("Enter requirement details. Type 'done' to finish, 'skip' to skip.\n")

    count = 0

    while True:
        print(f"\n--- Requirement #{len(rfp.requirements) + 1} ---")

        # Get requirement text
        print("Enter requirement text (or 'done' to finish):")
        text_lines = []
        while True:
            line = input()
            if line.lower() == 'done':
                if not text_lines:
                    # Save and exit
                    rfp.save(str(get_ground_truth_path(rfp_id)))
                    print(f"\nAdded {count} requirements. Total: {len(rfp.requirements)}")
                    return 0
                break
            if line.lower() == 'skip':
                break
            text_lines.append(line)

        if not text_lines:
            continue

        text = '\n'.join(text_lines)

        # Get section
        section = input("RFP Section (L/M/C/B/F/etc.): ").strip().upper()

        # Get subsection
        subsection = input("RFP Subsection (e.g., L.4.B.2): ").strip()

        # Get binding level
        print("Binding Level:")
        print("  1. Mandatory (shall/must)")
        print("  2. Highly Desirable (should)")
        print("  3. Desirable (may)")
        print("  4. Informational")
        binding_choice = input("Choice [1-4]: ").strip()
        binding_map = {
            "1": "Mandatory",
            "2": "Highly Desirable",
            "3": "Desirable",
            "4": "Informational",
        }
        binding = binding_map.get(binding_choice, "Mandatory")

        # Get binding keyword
        binding_keyword = input("Binding keyword (shall/must/should/may): ").strip().lower()

        # Get category
        print("Category:")
        print("  1. L_COMPLIANCE (Section L instructions)")
        print("  2. TECHNICAL (Section C/PWS/SOW)")
        print("  3. EVALUATION (Section M)")
        print("  4. ADMINISTRATIVE (Other sections)")
        print("  5. ATTACHMENT (From attachments)")
        cat_choice = input("Choice [1-5]: ").strip()
        cat_map = {
            "1": "L_COMPLIANCE",
            "2": "TECHNICAL",
            "3": "EVALUATION",
            "4": "ADMINISTRATIVE",
            "5": "ATTACHMENT",
        }
        category = cat_map.get(cat_choice, "TECHNICAL")

        # Get page number
        page_str = input("Page number: ").strip()
        page = int(page_str) if page_str.isdigit() else 0

        # Get source document
        source_doc = input("Source document (filename): ").strip()

        # Create requirement
        req_id = f"GT-{rfp_id[:10]}-{len(rfp.requirements) + 1:03d}"

        req = GroundTruthRequirement(
            gt_id=req_id,
            rfp_id=rfp_id,
            text=text,
            rfp_section=section,
            rfp_subsection=subsection,
            binding_level=binding,
            binding_keyword=binding_keyword,
            category=category,
            page_number=page,
            source_document=source_doc,
            annotator_id=os.environ.get("USER", "unknown"),
            annotation_timestamp=datetime.now().isoformat(),
            annotation_status="draft",
        )

        rfp.requirements.append(req)
        count += 1

        print(f"\nAdded: {req_id}")
        print(f"  Text: {text[:60]}...")

        # Save after each requirement
        rfp.save(str(get_ground_truth_path(rfp_id)))


def cmd_list(args):
    """List requirements in a ground truth RFP"""
    rfp = load_or_create_rfp(args.rfp_id)
    if not rfp:
        print(f"Error: RFP {args.rfp_id} not found")
        return 1

    print(f"\nGround Truth: {rfp.rfp_id}")
    print(f"Solicitation: {rfp.solicitation_number}")
    print(f"Agency: {rfp.agency}")
    print(f"Requirements: {len(rfp.requirements)}")
    print("\n" + "=" * 80)

    for i, req in enumerate(rfp.requirements, 1):
        status_icon = {
            "draft": "[ ]",
            "reviewed": "[R]",
            "approved": "[✓]",
            "disputed": "[!]",
        }.get(req.annotation_status, "[ ]")

        binding_abbrev = {
            "Mandatory": "M",
            "Highly Desirable": "HD",
            "Desirable": "D",
            "Informational": "I",
        }.get(req.binding_level, "?")

        print(f"\n{status_icon} {req.gt_id} | Section {req.rfp_section} | {binding_abbrev} | p.{req.page_number}")
        print(f"   {req.text[:100]}{'...' if len(req.text) > 100 else ''}")

    return 0


def cmd_stats(args):
    """Show statistics for a ground truth RFP"""
    rfp = load_or_create_rfp(args.rfp_id)
    if not rfp:
        print(f"Error: RFP {args.rfp_id} not found")
        return 1

    rfp.compute_stats()
    stats = rfp.stats

    print(f"\nGround Truth Statistics: {rfp.rfp_id}")
    print("=" * 60)
    print(f"Total Requirements: {stats['total_requirements']}")
    print(f"Compound Requirements: {stats['compound_requirements']}")
    print(f"With Cross-References: {stats['with_cross_references']}")

    print(f"\nBy Section:")
    for section, count in sorted(stats.get('by_section', {}).items()):
        print(f"  {section}: {count}")

    print(f"\nBy Binding Level:")
    for binding, count in sorted(stats.get('by_binding_level', {}).items()):
        print(f"  {binding}: {count}")

    print(f"\nBy Category:")
    for cat, count in sorted(stats.get('by_category', {}).items()):
        print(f"  {cat}: {count}")

    # Annotation status
    status_counts = {}
    for req in rfp.requirements:
        status = req.annotation_status
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"\nAnnotation Status:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    return 0


def cmd_export(args):
    """Export ground truth to specified format"""
    rfp = load_or_create_rfp(args.rfp_id)
    if not rfp:
        print(f"Error: RFP {args.rfp_id} not found")
        return 1

    rfp.compute_stats()

    if args.format == "json":
        output = json.dumps(rfp.to_dict(), indent=2)
    elif args.format == "csv":
        lines = ["gt_id,rfp_section,binding_level,category,page,text"]
        for req in rfp.requirements:
            text_escaped = req.text.replace('"', '""')
            lines.append(f'"{req.gt_id}","{req.rfp_section}","{req.binding_level}","{req.category}",{req.page_number},"{text_escaped}"')
        output = "\n".join(lines)
    else:
        print(f"Error: Unknown format {args.format}")
        return 1

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Exported to {args.output}")
    else:
        print(output)

    return 0


def cmd_approve(args):
    """Mark requirements as approved"""
    rfp = load_or_create_rfp(args.rfp_id)
    if not rfp:
        print(f"Error: RFP {args.rfp_id} not found")
        return 1

    approved_count = 0
    for req in rfp.requirements:
        if args.all or req.gt_id in args.ids:
            if req.annotation_status == "draft" or req.annotation_status == "reviewed":
                req.annotation_status = "approved"
                req.updated_at = datetime.now().isoformat()
                approved_count += 1

    if approved_count > 0:
        rfp.save(str(get_ground_truth_path(args.rfp_id)))
        print(f"Approved {approved_count} requirements")
    else:
        print("No requirements to approve")

    return 0


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(
        description="PropelAI Ground Truth Annotation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Create new ground truth:
    python -m validation.annotation_tool.cli new NIH_75N96025R00004 75N96025R00004 --agency NIH

  Add requirements interactively:
    python -m validation.annotation_tool.cli add NIH_75N96025R00004 --interactive

  List requirements:
    python -m validation.annotation_tool.cli list NIH_75N96025R00004

  Show statistics:
    python -m validation.annotation_tool.cli stats NIH_75N96025R00004

  Export to CSV:
    python -m validation.annotation_tool.cli export NIH_75N96025R00004 --format csv -o requirements.csv
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # new command
    new_parser = subparsers.add_parser("new", help="Create new ground truth RFP")
    new_parser.add_argument("rfp_id", help="Unique RFP identifier")
    new_parser.add_argument("solicitation_number", help="Official solicitation number")
    new_parser.add_argument("--agency", help="Agency (NIH, DoD, GSA, etc.)")
    new_parser.add_argument("--type", help="RFP type (Full-and-Open, IDIQ, BPA)")
    new_parser.add_argument("--format", help="Document format (UCF_STANDARD, GSA_SCHEDULE)")
    new_parser.add_argument("--title", help="RFP title")
    new_parser.add_argument("--annotator", help="Annotator ID")

    # add command
    add_parser = subparsers.add_parser("add", help="Add requirement to ground truth")
    add_parser.add_argument("rfp_id", help="RFP identifier")
    add_parser.add_argument("--text", help="Requirement text")
    add_parser.add_argument("--section", help="RFP section (L, M, C, etc.)")
    add_parser.add_argument("--binding", help="Binding level")
    add_parser.add_argument("--category", help="Requirement category")
    add_parser.add_argument("--page", type=int, help="Page number")
    add_parser.add_argument("--annotator", help="Annotator ID")
    add_parser.add_argument("--interactive", "-i", action="store_true",
                           help="Interactive annotation mode")

    # list command
    list_parser = subparsers.add_parser("list", help="List requirements")
    list_parser.add_argument("rfp_id", help="RFP identifier")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("rfp_id", help="RFP identifier")

    # export command
    export_parser = subparsers.add_parser("export", help="Export ground truth")
    export_parser.add_argument("rfp_id", help="RFP identifier")
    export_parser.add_argument("--format", "-f", choices=["json", "csv"], default="json",
                              help="Export format")
    export_parser.add_argument("--output", "-o", help="Output file (stdout if not specified)")

    # approve command
    approve_parser = subparsers.add_parser("approve", help="Approve requirements")
    approve_parser.add_argument("rfp_id", help="RFP identifier")
    approve_parser.add_argument("--all", action="store_true", help="Approve all draft/reviewed")
    approve_parser.add_argument("--ids", nargs="+", help="Specific requirement IDs to approve")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "new": cmd_new,
        "add": cmd_add,
        "list": cmd_list,
        "stats": cmd_stats,
        "export": cmd_export,
        "approve": cmd_approve,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
