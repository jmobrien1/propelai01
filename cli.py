#!/usr/bin/env python3
"""
PropelAI CLI - Command Line Interface
=====================================

Commands:
  propelai new <name>              Create a new proposal
  propelai upload <id> <file>      Upload RFP document
  propelai shred <id>              Run compliance shredding
  propelai strategy <id>           Generate win strategy
  propelai draft <id>              Generate draft content
  propelai redteam <id>            Run red team evaluation
  propelai status <id>             Show proposal status
  propelai export <id>             Export compliance matrix
  propelai serve                   Start the API server
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from core.state import create_initial_state, ProposalPhase
from agents.compliance_agent import create_compliance_agent
from agents.strategy_agent import create_strategy_agent
from agents.drafting_agent import create_drafting_agent
from agents.red_team_agent import create_red_team_agent
from tools.document_tools import create_document_loader, create_compliance_exporter

# Simple in-memory store for CLI (would use file/db in production)
PROPOSALS_FILE = os.path.expanduser("~/.propelai_proposals.json")


def get_console():
    """Get Rich console or fallback"""
    if RICH_AVAILABLE:
        return Console()
    return None


def print_output(message, style=None):
    """Print output with optional styling"""
    console = get_console()
    if console and style:
        console.print(message, style=style)
    else:
        print(message)


def load_proposals():
    """Load proposals from file"""
    if os.path.exists(PROPOSALS_FILE):
        with open(PROPOSALS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_proposals(proposals):
    """Save proposals to file"""
    with open(PROPOSALS_FILE, 'w') as f:
        json.dump(proposals, f, indent=2, default=str)


def get_proposal(proposal_id):
    """Get a proposal by ID"""
    proposals = load_proposals()
    if proposal_id not in proposals:
        print_output(f"Error: Proposal '{proposal_id}' not found", "red")
        sys.exit(1)
    return proposals[proposal_id]


def update_proposal(proposal_id, updates):
    """Update a proposal"""
    proposals = load_proposals()
    if proposal_id in proposals:
        for key, value in updates.items():
            if isinstance(value, list) and isinstance(proposals[proposal_id].get(key), list):
                proposals[proposal_id][key] = proposals[proposal_id][key] + value
            else:
                proposals[proposal_id][key] = value
        proposals[proposal_id]["updated_at"] = datetime.now().isoformat()
    save_proposals(proposals)
    return proposals[proposal_id]


def cmd_new(args):
    """Create a new proposal"""
    import uuid
    
    proposal_id = f"PROP-{uuid.uuid4().hex[:8].upper()}"
    
    state = create_initial_state(
        proposal_id=proposal_id,
        client_name=args.client or "Client",
        opportunity_name=args.name,
        solicitation_number=args.solicitation or "TBD",
        due_date=args.due_date
    )
    
    proposals = load_proposals()
    proposals[proposal_id] = state
    save_proposals(proposals)
    
    print_output(f"✓ Created proposal: {proposal_id}", "green")
    print_output(f"  Name: {args.name}")
    print_output(f"  Client: {state['client_name']}")
    
    return proposal_id


def cmd_upload(args):
    """Upload an RFP document"""
    proposal = get_proposal(args.proposal_id)
    
    if not os.path.exists(args.file):
        print_output(f"Error: File not found: {args.file}", "red")
        sys.exit(1)
    
    print_output(f"Uploading {args.file}...", "yellow")
    
    loader = create_document_loader()
    parsed = loader.load(args.file)
    
    update_proposal(args.proposal_id, {
        "rfp_raw_text": parsed.raw_text,
        "rfp_file_paths": proposal.get("rfp_file_paths", []) + [args.file],
        "rfp_metadata": {
            **proposal.get("rfp_metadata", {}),
            "structure": parsed.structure
        }
    })
    
    print_output(f"✓ Uploaded: {parsed.file_name}", "green")
    print_output(f"  Pages: {parsed.total_pages}")
    print_output(f"  Characters: {parsed.total_chars:,}")


def cmd_shred(args):
    """Run compliance shredding"""
    proposal = get_proposal(args.proposal_id)
    
    if not proposal.get("rfp_raw_text"):
        print_output("Error: No RFP document uploaded. Run 'upload' first.", "red")
        sys.exit(1)
    
    print_output("Running Compliance Agent (Shred)...", "yellow")
    
    agent = create_compliance_agent()
    result = agent(proposal)
    
    update_proposal(args.proposal_id, result)
    
    print_output("✓ Shredding complete", "green")
    print_output(f"  Requirements: {len(result.get('requirements', []))}")
    print_output(f"  Instructions: {len(result.get('instructions', []))}")
    print_output(f"  Eval Criteria: {len(result.get('evaluation_criteria', []))}")


def cmd_strategy(args):
    """Generate win strategy"""
    proposal = get_proposal(args.proposal_id)
    
    if not proposal.get("evaluation_criteria"):
        print_output("Error: Run 'shred' first to extract evaluation criteria.", "red")
        sys.exit(1)
    
    print_output("Running Strategy Agent...", "yellow")
    
    agent = create_strategy_agent()
    result = agent(proposal)
    
    update_proposal(args.proposal_id, result)
    
    print_output("✓ Strategy complete", "green")
    print_output(f"  Win Themes: {len(result.get('win_themes', []))}")
    
    for theme in result.get("win_themes", []):
        print_output(f"\n  [{theme['id']}] {theme['theme_text']}")
        print_output(f"    Discriminator: {theme['discriminator']}", "dim")


def cmd_draft(args):
    """Generate draft content"""
    proposal = get_proposal(args.proposal_id)
    
    if not proposal.get("annotated_outline"):
        print_output("Error: Run 'strategy' first to generate outline.", "red")
        sys.exit(1)
    
    print_output("Running Drafting Agent...", "yellow")
    
    agent = create_drafting_agent()
    result = agent(proposal)
    
    update_proposal(args.proposal_id, result)
    
    drafts = result.get("draft_sections", {})
    total_words = sum(d.get("word_count", 0) for d in drafts.values())
    uncited = sum(len(d.get("uncited_claims", [])) for d in drafts.values())
    
    print_output("✓ Drafting complete", "green")
    print_output(f"  Sections: {len(drafts)}")
    print_output(f"  Total Words: {total_words:,}")
    
    if uncited > 0:
        print_output(f"  ⚠️  Uncited Claims: {uncited} (HIGH RISK)", "yellow")


def cmd_redteam(args):
    """Run red team evaluation"""
    proposal = get_proposal(args.proposal_id)
    
    if not proposal.get("draft_sections"):
        print_output("Error: Run 'draft' first to generate content.", "red")
        sys.exit(1)
    
    print_output("Running Red Team Agent...", "yellow")
    
    agent = create_red_team_agent()
    result = agent(proposal)
    
    update_proposal(args.proposal_id, result)
    
    feedback = result.get("red_team_feedback", [{}])[-1]
    
    color_display = {
        "blue": "BLUE (Exceptional)",
        "green": "GREEN (Acceptable)",
        "yellow": "YELLOW (Marginal)",
        "red": "RED (Unacceptable)"
    }
    
    print_output("✓ Red Team evaluation complete", "green")
    print_output(f"\n  Overall Score: {color_display.get(feedback.get('overall_score', ''), 'Unknown')}")
    print_output(f"  Numeric Score: {feedback.get('overall_numeric', 0):.1f}/100")
    print_output(f"  Recommendation: {feedback.get('recommendation', 'unknown').replace('_', ' ').title()}")
    
    deficiencies = feedback.get("critical_deficiencies", [])
    if deficiencies:
        print_output(f"\n  ⚠️  Critical Deficiencies: {len(deficiencies)}", "yellow")
        for d in deficiencies[:3]:
            print_output(f"    • {d.get('description', '')[:80]}")


def cmd_status(args):
    """Show proposal status"""
    proposal = get_proposal(args.proposal_id)
    
    console = get_console()
    
    if console and RICH_AVAILABLE:
        table = Table(title=f"Proposal: {args.proposal_id}", box=box.ROUNDED)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Client", proposal.get("client_name", "N/A"))
        table.add_row("Opportunity", proposal.get("opportunity_name", "N/A"))
        table.add_row("Solicitation", proposal.get("solicitation_number", "N/A"))
        table.add_row("Phase", proposal.get("current_phase", "N/A"))
        table.add_row("Due Date", proposal.get("due_date", "N/A"))
        table.add_row("Requirements", str(len(proposal.get("requirements", []))))
        table.add_row("Win Themes", str(len(proposal.get("win_themes", []))))
        table.add_row("Draft Sections", str(len(proposal.get("draft_sections", {}))))
        table.add_row("Quality Score", f"{proposal.get('proposal_quality_score', 0):.1f}")
        table.add_row("Updated", proposal.get("updated_at", "N/A")[:19])
        
        console.print(table)
    else:
        print(f"\nProposal: {args.proposal_id}")
        print(f"  Client: {proposal.get('client_name', 'N/A')}")
        print(f"  Opportunity: {proposal.get('opportunity_name', 'N/A')}")
        print(f"  Phase: {proposal.get('current_phase', 'N/A')}")
        print(f"  Requirements: {len(proposal.get('requirements', []))}")
        print(f"  Quality Score: {proposal.get('proposal_quality_score', 0):.1f}")


def cmd_export(args):
    """Export compliance matrix"""
    proposal = get_proposal(args.proposal_id)
    
    matrix = proposal.get("compliance_matrix", [])
    if not matrix:
        print_output("Error: No compliance matrix available. Run 'shred' first.", "red")
        sys.exit(1)
    
    output_path = args.output or f"{args.proposal_id}_compliance_matrix.xlsx"
    
    exporter = create_compliance_exporter()
    result_path = exporter.export(matrix, output_path)
    
    print_output(f"✓ Exported to: {result_path}", "green")


def cmd_list(args):
    """List all proposals"""
    proposals = load_proposals()
    
    if not proposals:
        print_output("No proposals found.", "yellow")
        return
    
    console = get_console()
    
    if console and RICH_AVAILABLE:
        table = Table(title="Proposals", box=box.ROUNDED)
        table.add_column("ID", style="cyan")
        table.add_column("Opportunity", style="white")
        table.add_column("Phase", style="green")
        table.add_column("Score", style="yellow")
        
        for pid, p in proposals.items():
            table.add_row(
                pid,
                p.get("opportunity_name", "N/A")[:40],
                p.get("current_phase", "N/A"),
                f"{p.get('proposal_quality_score', 0):.0f}"
            )
        
        console.print(table)
    else:
        print("\nProposals:")
        for pid, p in proposals.items():
            print(f"  {pid}: {p.get('opportunity_name', 'N/A')[:40]} [{p.get('current_phase', 'N/A')}]")


def cmd_serve(args):
    """Start the API server"""
    import uvicorn
    from api.main import app
    
    print_output(f"Starting PropelAI API server on {args.host}:{args.port}...", "cyan")
    uvicorn.run(app, host=args.host, port=args.port)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="PropelAI - Autonomous Proposal Operating System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  propelai new "DOT Portal Modernization" --client "ACME Corp"
  propelai upload PROP-ABC12345 ./rfp.pdf
  propelai shred PROP-ABC12345
  propelai strategy PROP-ABC12345
  propelai draft PROP-ABC12345
  propelai redteam PROP-ABC12345
  propelai status PROP-ABC12345
  propelai export PROP-ABC12345
  propelai serve
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # new command
    new_parser = subparsers.add_parser("new", help="Create a new proposal")
    new_parser.add_argument("name", help="Opportunity name")
    new_parser.add_argument("--client", "-c", help="Client name")
    new_parser.add_argument("--solicitation", "-s", help="Solicitation number")
    new_parser.add_argument("--due-date", "-d", help="Due date (YYYY-MM-DD)")
    
    # upload command
    upload_parser = subparsers.add_parser("upload", help="Upload RFP document")
    upload_parser.add_argument("proposal_id", help="Proposal ID")
    upload_parser.add_argument("file", help="Path to RFP document")
    
    # shred command
    shred_parser = subparsers.add_parser("shred", help="Run compliance shredding")
    shred_parser.add_argument("proposal_id", help="Proposal ID")
    
    # strategy command
    strategy_parser = subparsers.add_parser("strategy", help="Generate win strategy")
    strategy_parser.add_argument("proposal_id", help="Proposal ID")
    
    # draft command
    draft_parser = subparsers.add_parser("draft", help="Generate draft content")
    draft_parser.add_argument("proposal_id", help="Proposal ID")
    
    # redteam command
    redteam_parser = subparsers.add_parser("redteam", help="Run red team evaluation")
    redteam_parser.add_argument("proposal_id", help="Proposal ID")
    
    # status command
    status_parser = subparsers.add_parser("status", help="Show proposal status")
    status_parser.add_argument("proposal_id", help="Proposal ID")
    
    # export command
    export_parser = subparsers.add_parser("export", help="Export compliance matrix")
    export_parser.add_argument("proposal_id", help="Proposal ID")
    export_parser.add_argument("--output", "-o", help="Output file path")
    
    # list command
    list_parser = subparsers.add_parser("list", help="List all proposals")
    
    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", "-p", type=int, default=8000, help="Port to bind")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to command handler
    commands = {
        "new": cmd_new,
        "upload": cmd_upload,
        "shred": cmd_shred,
        "strategy": cmd_strategy,
        "draft": cmd_draft,
        "redteam": cmd_redteam,
        "status": cmd_status,
        "export": cmd_export,
        "list": cmd_list,
        "serve": cmd_serve,
    }
    
    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
