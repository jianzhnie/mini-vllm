#!/usr/bin/env python3
"""Test runner script for mini-vLLM tests.

This script provides a convenient way to run tests with various options.

Examples:
    python run_tests.py                 # Run all tests
    python run_tests.py -v              # Verbose output
    python run_tests.py -k test_config  # Run specific test
    python run_tests.py --slow          # Include slow tests
    python run_tests.py --coverage      # Generate coverage report
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Run pytest with specified arguments."""
    parser = argparse.ArgumentParser(
        description='Run mini-vLLM tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py -v                 # Verbose output
  python run_tests.py -k test_config     # Run specific test pattern
  python run_tests.py -m integration     # Run integration tests only
  python run_tests.py --coverage         # Generate coverage report
  python run_tests.py --slow             # Include slow tests
        ''')

    parser.add_argument('-v',
                        '--verbose',
                        action='count',
                        default=0,
                        help='Increase verbosity level')
    parser.add_argument('-q',
                        '--quiet',
                        action='store_true',
                        help='Decrease verbosity (quiet mode)')
    parser.add_argument(
        '-k',
        '--keyword',
        type=str,
        help='Only run tests matching the given keyword expression')
    parser.add_argument(
        '-m',
        '--marker',
        type=str,
        help='Only run tests matching the given marker expression')
    parser.add_argument('--coverage',
                        action='store_true',
                        help='Generate coverage report')
    parser.add_argument('--slow',
                        action='store_true',
                        help='Include slow tests')
    parser.add_argument('--failed',
                        action='store_true',
                        help='Run only previously failed tests')
    parser.add_argument('--junit',
                        action='store_true',
                        help='Generate JUnit XML report')
    parser.add_argument('tests',
                        nargs='*',
                        help='Specific test files or directories to run')

    args = parser.parse_args()

    # Build pytest command
    cmd = ['pytest']

    # Add verbosity
    if args.quiet:
        cmd.append('-q')
    elif args.verbose:
        cmd.extend(['-v'] * args.verbose)

    # Add test selection
    if args.keyword:
        cmd.extend(['-k', args.keyword])

    if args.marker:
        cmd.extend(['-m', args.marker])
    elif not args.slow:
        # Exclude slow tests by default
        cmd.extend(['-m', 'not slow'])

    # Add coverage
    if args.coverage:
        cmd.extend([
            '--cov=minivllm', '--cov-report=html', '--cov-report=term-missing'
        ])

    # Add failed tests
    if args.failed:
        cmd.append('--lf')

    # Add JUnit report
    if args.junit:
        cmd.append('--junit-xml=test_results.xml')

    # Add specific tests or default to tests directory
    if args.tests:
        cmd.extend(args.tests)
    else:
        tests_dir = Path(__file__).parent
        cmd.append(str(tests_dir))

    # Run pytest
    return subprocess.call(cmd)


if __name__ == '__main__':
    sys.exit(main())
