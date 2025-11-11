"""
Run all tests for the nn-from-scratch project.

Usage:
    python tests/run_all_tests.py [semester] [lab]
    
Examples:
    python tests/run_all_tests.py                  # Run all tests
    python tests/run_all_tests.py 1                # Run semester 1 tests
    python tests/run_all_tests.py 1 1              # Run semester 1, lab 1 tests
"""

import sys
import subprocess
from pathlib import Path


def run_tests(semester=None, lab=None):
    """Run tests with optional filtering."""
    
    # Base command
    cmd = ["pytest", "-v", "--tb=short"]
    
    # Determine test path
    if semester and lab:
        test_path = f"semester{semester}/lab{lab}_*/tests/"
        print(f"Running tests for Semester {semester}, Lab {lab}...")
    elif semester:
        test_path = f"semester{semester}/*/tests/"
        print(f"Running tests for Semester {semester}...")
    else:
        test_path = "semester*/*/tests/"
        print("Running all tests...")
    
    cmd.append(test_path)
    
    # Add coverage if available
    try:
        import pytest_cov
        cmd.extend(["--cov=semester1", "--cov=semester2", "--cov=semester3",
                   "--cov=semester4", "--cov=semester5", "--cov=semester6",
                   "--cov-report=term-missing"])
    except ImportError:
        print("Note: Install pytest-cov for coverage reports")
    
    # Run tests
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main entry point."""
    semester = None
    lab = None
    
    if len(sys.argv) > 1:
        semester = sys.argv[1]
    
    if len(sys.argv) > 2:
        lab = sys.argv[2]
    
    exit_code = run_tests(semester, lab)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
