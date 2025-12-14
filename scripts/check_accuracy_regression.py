#!/usr/bin/env python3
"""
Check for accuracy regression between current and baseline metrics.

Usage:
    python scripts/check_accuracy_regression.py \
        --current accuracy_report.json \
        --baseline .accuracy_baseline.json \
        --threshold 0.02

Exit codes:
    0: No regression detected
    1: Regression detected
    2: Error (missing files, invalid JSON, etc.)
"""

import json
import sys
import argparse
from typing import Dict, Any, List, Tuple


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file"""
    try:
        with open(filepath) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(2)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filepath}: {e}")
        sys.exit(2)


def check_regression(
    current: Dict[str, Any],
    baseline: Dict[str, Any],
    threshold: float
) -> Tuple[bool, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Check if current metrics show regression from baseline.

    Returns:
        Tuple of (has_regression, regressions, improvements)
    """
    regressions = []
    improvements = []

    # Metrics where decrease is bad
    decrease_metrics = [
        'precision',
        'recall',
        'f1_score',
        'section_accuracy',
        'binding_accuracy',
        'mandatory_recall',
        'category_accuracy',
    ]

    # Metrics where increase is bad
    increase_metrics = [
        'unknown_section_rate',
        'false_positive_rate',
        'critical_miss_rate',
    ]

    # Check decrease metrics
    for metric in decrease_metrics:
        current_val = current.get(metric)
        baseline_val = baseline.get(metric)

        if current_val is None or baseline_val is None:
            continue

        diff = baseline_val - current_val

        if diff > threshold:
            regressions.append({
                'metric': metric,
                'baseline': baseline_val,
                'current': current_val,
                'diff': diff,
                'direction': 'decreased'
            })
        elif diff < -threshold:
            improvements.append({
                'metric': metric,
                'baseline': baseline_val,
                'current': current_val,
                'improvement': -diff,
            })

    # Check increase metrics
    for metric in increase_metrics:
        current_val = current.get(metric)
        baseline_val = baseline.get(metric)

        if current_val is None or baseline_val is None:
            continue

        diff = current_val - baseline_val

        if diff > threshold:
            regressions.append({
                'metric': metric,
                'baseline': baseline_val,
                'current': current_val,
                'diff': diff,
                'direction': 'increased'
            })
        elif diff < -threshold:
            improvements.append({
                'metric': metric,
                'baseline': baseline_val,
                'current': current_val,
                'improvement': -diff,
            })

    return (len(regressions) > 0, regressions, improvements)


def main():
    parser = argparse.ArgumentParser(
        description="Check for accuracy regression"
    )
    parser.add_argument(
        '--current', '-c',
        required=True,
        help='Path to current accuracy report JSON'
    )
    parser.add_argument(
        '--baseline', '-b',
        required=True,
        help='Path to baseline accuracy JSON'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.02,
        help='Maximum acceptable regression (default: 0.02 = 2%%)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file for comparison results'
    )

    args = parser.parse_args()

    # Load files
    current = load_json(args.current)
    baseline = load_json(args.baseline)

    # Check regression
    has_regression, regressions, improvements = check_regression(
        current, baseline, args.threshold
    )

    # Print results
    print("=" * 60)
    print("ACCURACY REGRESSION CHECK")
    print("=" * 60)
    print(f"Threshold: {args.threshold:.1%}")
    print()

    if improvements:
        print("IMPROVEMENTS:")
        for imp in improvements:
            print(f"  {imp['metric']}: {imp['baseline']:.3f} -> {imp['current']:.3f} (+{imp['improvement']:.3f})")
        print()

    if regressions:
        print("REGRESSIONS DETECTED:")
        for reg in regressions:
            print(f"  {reg['metric']}: {reg['baseline']:.3f} -> {reg['current']:.3f} ({reg['direction']} by {reg['diff']:.3f})")
        print()
        print("STATUS: FAIL")
    else:
        print("STATUS: PASS - No regressions detected")

    print("=" * 60)

    # Save output if requested
    if args.output:
        result = {
            'has_regression': has_regression,
            'threshold': args.threshold,
            'regressions': regressions,
            'improvements': improvements,
        }
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")

    # Exit with appropriate code
    sys.exit(1 if has_regression else 0)


if __name__ == '__main__':
    main()
