#!/usr/bin/env python3
"""
PropelAI APOS - Full Workflow Demo
==================================

This script demonstrates the complete proposal generation workflow:
1. Create a new proposal
2. Upload/ingest an RFP document
3. Run the Compliance Agent (Shred)
4. Run the Strategy Agent (Win Themes)
5. Run the Drafting Agent (Content Generation)
6. Run the Red Team Agent (Evaluation)
7. Review results and export

This simulates "Brenda's" workflow from the PRD.
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich import box

# Import PropelAI components (avoiding orchestrator which needs langgraph)
from core.state import create_initial_state, ProposalPhase
from agents.compliance_agent import create_compliance_agent
from agents.strategy_agent import create_strategy_agent
from agents.drafting_agent import create_drafting_agent, create_research_agent
from agents.red_team_agent import create_red_team_agent

console = Console()


# ============== Sample RFP Content ==============

SAMPLE_RFP = """
SOLICITATION NUMBER: GS-00F-12345
FEDERAL ACQUISITION SERVICE
REQUEST FOR PROPOSAL

SECTION C - STATEMENT OF WORK

C.1 BACKGROUND
The Department of Technology Services (DTS) requires modernization of its legacy 
business permitting system to improve citizen services and operational efficiency.

C.2 SCOPE OF WORK
The Contractor shall provide all labor, materials, equipment, and supervision 
necessary to design, develop, implement, and maintain a modern cloud-based 
Business Permitting Portal.

C.3 TECHNICAL REQUIREMENTS

C.3.1 System Architecture
The Contractor shall implement a cloud-native architecture using microservices.
The system shall support a minimum of 10,000 concurrent users.
The system shall achieve 99.9% uptime availability.

C.3.2 User Interface
The Contractor shall develop a responsive web application accessible on desktop and mobile devices.
The system shall comply with Section 508 accessibility requirements.
The system shall support multiple languages including English and Spanish.

C.3.3 Integration Requirements
The Contractor shall integrate with the existing SAP financial system.
The Contractor shall provide APIs for third-party system integration.
The system shall support single sign-on (SSO) using SAML 2.0.

C.3.4 Security Requirements
The Contractor shall implement FedRAMP Moderate security controls.
All data shall be encrypted at rest and in transit using FIPS 140-2 validated encryption.
The Contractor shall conduct annual penetration testing.

C.4 DELIVERABLES
The Contractor shall provide the following deliverables:
- System Design Document (SDD)
- User Training Materials
- Operations and Maintenance Manual
- Monthly Status Reports

SECTION L - INSTRUCTIONS TO OFFERORS

L.1 PROPOSAL FORMAT
Proposals shall be submitted in the following volumes:
- Volume I: Technical Approach (50 pages maximum)
- Volume II: Management Approach (25 pages maximum)  
- Volume III: Past Performance (15 pages maximum)
- Volume IV: Price/Cost (no page limit)

L.2 FORMAT REQUIREMENTS
- Font: Times New Roman, 12-point minimum
- Margins: 1 inch on all sides
- Single-spaced text

L.3 SUBMISSION REQUIREMENTS
Proposals must be submitted electronically via the agency portal by 2:00 PM EST on the due date.

SECTION M - EVALUATION CRITERIA

M.1 EVALUATION FACTORS
Proposals will be evaluated based on the following factors in descending order of importance:

M.1.1 Technical Approach (Most Important)
The Government will evaluate the Offeror's understanding of requirements and proposed 
technical solution. Innovation and use of modern technologies will be favorably considered.

M.1.2 Past Performance (Important)
The Government will evaluate the Offeror's record of performance on contracts of 
similar size, scope, and complexity. Contracts with state and local government 
agencies are highly relevant.

M.1.3 Management Approach (Important)
The Government will evaluate the Offeror's management plan, including staffing, 
quality assurance, and risk mitigation approaches.

M.1.4 Price/Cost (Less Important than Non-Price Factors)
Price will be evaluated for reasonableness and realism. The Government intends to 
make award on a best value basis.

M.2 EVALUATION METHODOLOGY
The Government will use color/adjectival ratings:
- Blue: Exceptional
- Green: Acceptable  
- Yellow: Marginal
- Red: Unacceptable
"""


def print_header():
    """Print the PropelAI header"""
    console.print(Panel.fit(
        "[bold blue]PropelAI[/bold blue] [white]Autonomous Proposal Operating System[/white]\n"
        "[dim]Cycle 1-4 Demo: Full Proposal Workflow[/dim]",
        border_style="blue"
    ))
    console.print()


def print_phase(phase: str, description: str):
    """Print a phase header"""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]PHASE: {phase}[/bold cyan]")
    console.print(f"[dim]{description}[/dim]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")


def print_agent_result(agent_name: str, result: Dict[str, Any]):
    """Print agent execution result"""
    console.print(f"\n[bold green]✓ {agent_name} Complete[/bold green]")
    
    # Show trace log
    trace_log = result.get("agent_trace_log", [])
    if trace_log:
        latest = trace_log[-1]
        console.print(f"  [dim]Action: {latest.get('action', 'N/A')}[/dim]")
        console.print(f"  [dim]Output: {latest.get('output_summary', 'N/A')}[/dim]")
        if latest.get('duration_ms'):
            console.print(f"  [dim]Duration: {latest['duration_ms']}ms[/dim]")


def demo_workflow():
    """Run the full PropelAI workflow demo"""
    
    print_header()
    
    # ========== STEP 1: Create Proposal ==========
    print_phase("1. PROPOSAL CREATION", "Initialize a new proposal engagement")
    
    console.print("[yellow]Creating new proposal...[/yellow]")
    
    state = create_initial_state(
        proposal_id="PROP-DEMO-001",
        client_name="TechServices Inc.",
        opportunity_name="DTS Business Permitting Portal Modernization",
        solicitation_number="GS-00F-12345",
        due_date="2025-02-15"
    )
    
    console.print(f"[green]✓ Proposal Created: {state['proposal_id']}[/green]")
    console.print(f"  Client: {state['client_name']}")
    console.print(f"  Opportunity: {state['opportunity_name']}")
    console.print(f"  Due Date: {state['due_date']}")
    
    # ========== STEP 2: Ingest RFP ==========
    print_phase("2. RFP INGESTION", "Load the RFP document into the system")
    
    console.print("[yellow]Ingesting RFP document...[/yellow]")
    
    # Simulate document upload
    state["rfp_raw_text"] = SAMPLE_RFP
    state["rfp_file_paths"] = ["/uploads/DEMO/RFP_GS-00F-12345.pdf"]
    
    console.print(f"[green]✓ RFP Ingested[/green]")
    console.print(f"  Characters: {len(SAMPLE_RFP):,}")
    console.print(f"  Sections detected: C, L, M")
    
    # ========== STEP 3: Compliance Agent (Shred) ==========
    print_phase("3. INTELLIGENT SHRED", "Compliance Agent extracts requirements and builds matrix")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Running Compliance Agent...", total=None)
        
        compliance_agent = create_compliance_agent()
        compliance_result = compliance_agent(state)
        
        # Update state
        state.update(compliance_result)
        progress.remove_task(task)
    
    print_agent_result("Compliance Agent (The Paralegal)", compliance_result)
    
    # Show requirements summary
    requirements = state.get("requirements", [])
    eval_criteria = state.get("evaluation_criteria", [])
    instructions = state.get("instructions", [])
    
    table = Table(title="Shred Results", box=box.ROUNDED)
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Sample", style="dim")
    
    table.add_row(
        "Requirements (Section C)",
        str(len(requirements)),
        requirements[0]["text"][:50] + "..." if requirements else "N/A"
    )
    table.add_row(
        "Instructions (Section L)",
        str(len(instructions)),
        instructions[0]["text"][:50] + "..." if instructions else "N/A"
    )
    table.add_row(
        "Eval Criteria (Section M)",
        str(len(eval_criteria)),
        eval_criteria[0]["factor_name"] if eval_criteria else "N/A"
    )
    
    console.print(table)
    
    # Show compliance matrix preview
    matrix = state.get("compliance_matrix", [])
    if matrix:
        console.print(f"\n[bold]Compliance Matrix: {len(matrix)} items[/bold]")
        for item in matrix[:3]:
            console.print(f"  • [{item['requirement_id']}] {item['requirement_text'][:60]}...")
    
    # ========== STEP 4: Strategy Agent (Win Themes) ==========
    print_phase("4. STRATEGY ENGINE", "Strategy Agent develops win themes and storyboard")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Running Strategy Agent...", total=None)
        
        strategy_agent = create_strategy_agent()
        strategy_result = strategy_agent(state)
        
        state.update(strategy_result)
        progress.remove_task(task)
    
    print_agent_result("Strategy Agent (The Capture Manager)", strategy_result)
    
    # Show win themes
    win_themes = state.get("win_themes", [])
    if win_themes:
        console.print("\n[bold]Win Themes Developed:[/bold]")
        for theme in win_themes:
            console.print(f"\n  [bold cyan]{theme['id']}[/bold cyan]: {theme['theme_text']}")
            console.print(f"  [dim]Discriminator: {theme['discriminator']}[/dim]")
            if theme.get('ghosting_language'):
                console.print(f"  [dim italic]Ghost: {theme['ghosting_language'][:80]}...[/dim italic]")
    
    # Show outline
    outline = state.get("annotated_outline", {})
    if outline:
        console.print(f"\n[bold]Annotated Outline (Page Limit: {outline.get('total_page_limit', 'N/A')}):[/bold]")
        for vol_id, vol_data in outline.get("volumes", {}).items():
            console.print(f"\n  [cyan]{vol_data['title']}[/cyan] ({vol_data['page_allocation']} pages)")
            for section in vol_data.get("sections", [])[:2]:
                console.print(f"    • {section.get('section_number', '')}: {section.get('title', '')}")
    
    # ========== STEP 5: Drafting Agent (Content) ==========
    print_phase("5. COLLABORATIVE DRAFTING", "Drafting Agent generates citation-backed content")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Running Drafting Agent...", total=None)
        
        drafting_agent = create_drafting_agent()
        drafting_result = drafting_agent(state)
        
        state.update(drafting_result)
        progress.remove_task(task)
    
    print_agent_result("Drafting Agent (The Writer)", drafting_result)
    
    # Show draft summary
    drafts = state.get("draft_sections", {})
    total_words = sum(d.get("word_count", 0) for d in drafts.values())
    total_citations = sum(len(d.get("citations", [])) for d in drafts.values())
    total_uncited = sum(len(d.get("uncited_claims", [])) for d in drafts.values())
    
    table = Table(title="Draft Content Summary", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Sections Drafted", str(len(drafts)))
    table.add_row("Total Words", f"{total_words:,}")
    table.add_row("Citations", str(total_citations))
    table.add_row("Uncited Claims", str(total_uncited) + (" ⚠️ HIGH RISK" if total_uncited > 0 else " ✓"))
    
    console.print(table)
    
    # Show sample draft
    if drafts:
        first_draft = list(drafts.values())[0]
        console.print(f"\n[bold]Sample Draft ({first_draft.get('section_title', 'N/A')}):[/bold]")
        console.print(Panel(
            first_draft.get("content", "")[:500] + "...",
            title=f"Section {first_draft.get('section_id', '')}",
            border_style="dim"
        ))
    
    # ========== STEP 6: Red Team Agent (Evaluation) ==========
    print_phase("6. WAR ROOM", "Red Team Agent scores the proposal like a government evaluator")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Running Red Team Agent...", total=None)
        
        red_team_agent = create_red_team_agent()
        redteam_result = red_team_agent(state)
        
        state.update(redteam_result)
        progress.remove_task(task)
    
    print_agent_result("Red Team Agent (The Evaluator)", redteam_result)
    
    # Show evaluation results
    feedback = state.get("red_team_feedback", [{}])[-1]
    
    # Color mapping for display
    color_map = {
        "blue": "[bold blue]BLUE (Exceptional)[/bold blue]",
        "green": "[bold green]GREEN (Acceptable)[/bold green]",
        "yellow": "[bold yellow]YELLOW (Marginal)[/bold yellow]",
        "red": "[bold red]RED (Unacceptable)[/bold red]"
    }
    
    overall_color = feedback.get("overall_score", "unknown")
    overall_numeric = feedback.get("overall_numeric", 0)
    recommendation = feedback.get("recommendation", "unknown")
    
    console.print(Panel(
        f"Overall Score: {color_map.get(overall_color, overall_color)}\n"
        f"Numeric Score: {overall_numeric:.1f}/100\n"
        f"Recommendation: [bold]{recommendation.replace('_', ' ').title()}[/bold]",
        title="[bold]PROPOSAL EVALUATION RESULT[/bold]",
        border_style="cyan"
    ))
    
    # Show section scores
    section_scores = feedback.get("section_scores", [])
    if section_scores:
        table = Table(title="Section Scores", box=box.ROUNDED)
        table.add_column("Section", style="white")
        table.add_column("Score", style="cyan")
        table.add_column("Compliance", style="green")
        table.add_column("Issues", style="yellow")
        
        for section in section_scores:
            score_display = color_map.get(section.get("color_score", ""), section.get("color_score", ""))
            compliance = f"{section.get('compliance_rate', 0)*100:.0f}%"
            issues = f"S:{section.get('strengths_count', 0)} W:{section.get('weaknesses_count', 0)} D:{section.get('deficiencies_count', 0)}"
            table.add_row(
                section.get("section_title", "")[:30],
                score_display,
                compliance,
                issues
            )
        
        console.print(table)
    
    # Show critical deficiencies
    deficiencies = feedback.get("critical_deficiencies", [])
    if deficiencies:
        console.print("\n[bold red]⚠️ CRITICAL DEFICIENCIES:[/bold red]")
        for deficiency in deficiencies[:5]:
            console.print(f"  • {deficiency.get('description', '')}")
            if deficiency.get("remediation"):
                console.print(f"    [dim]Fix: {deficiency['remediation']}[/dim]")
    
    # ========== SUMMARY ==========
    print_phase("7. WORKFLOW COMPLETE", "Proposal ready for human review")
    
    # Final summary
    console.print(Panel(
        f"[bold]Proposal ID:[/bold] {state['proposal_id']}\n"
        f"[bold]Current Phase:[/bold] {state['current_phase']}\n"
        f"[bold]Requirements Extracted:[/bold] {len(state.get('requirements', []))}\n"
        f"[bold]Win Themes:[/bold] {len(state.get('win_themes', []))}\n"
        f"[bold]Sections Drafted:[/bold] {len(state.get('draft_sections', {}))}\n"
        f"[bold]Quality Score:[/bold] {state.get('proposal_quality_score', 0):.1f}/100\n"
        f"[bold]Agent Trace Entries:[/bold] {len(state.get('agent_trace_log', []))}\n\n"
        f"[dim]Next Step: Human-in-the-Loop review of {len(deficiencies)} deficiencies[/dim]",
        title="[bold cyan]PROPOSAL SUMMARY[/bold cyan]",
        border_style="green"
    ))
    
    # Audit log summary
    trace_log = state.get("agent_trace_log", [])
    console.print(f"\n[bold]Audit Log ({len(trace_log)} entries):[/bold]")
    for entry in trace_log:
        console.print(f"  [{entry.get('timestamp', '')[:19]}] {entry.get('agent_name', '')}: {entry.get('action', '')}")
    
    console.print("\n[bold green]✓ PropelAI APOS Demo Complete![/bold green]")
    console.print("[dim]The proposal is now ready for Brenda's review in the War Room.[/dim]\n")
    
    return state


if __name__ == "__main__":
    try:
        final_state = demo_workflow()
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise
