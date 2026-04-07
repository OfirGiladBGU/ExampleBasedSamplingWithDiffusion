"""
Setup and validation script for train_free_v2.

This script:
1. Checks dependencies
2. Validates directory structure
3. Runs unit tests
4. Provides quick-start instructions
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Check that required packages are installed."""
    print("=" * 80)
    print("Checking dependencies...")
    print("=" * 80)
    
    required = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
    }
    
    optional = {
        'matplotlib': 'matplotlib (optional, for visualization)',
        'tqdm': 'tqdm (optional, for progress bars)',
    }
    
    missing = []
    
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} (REQUIRED)")
            missing.append(name)
    
    for module, name in optional.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} (optional, install for best experience)")
    
    return len(missing) == 0


def check_structure():
    """Verify directory structure."""
    print("\n" + "=" * 80)
    print("Checking directory structure...")
    print("=" * 80)
    
    train_free_v2_dir = Path(__file__).parent
    required_files = [
        'sinkhorn_loss.py',
        'guided_sample_dps.py',
        'utils_guidance.py',
        'sample_dps.py',
        '__init__.py',
        'README.md',
        'sample_config_dps.json',
    ]
    
    required_dirs = [
        'examples',
        'tests',
        'sample_outputs',
    ]
    
    all_ok = True
    
    for filename in required_files:
        filepath = train_free_v2_dir / filename
        if filepath.exists():
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} (MISSING)")
            all_ok = False
    
    for dirname in required_dirs:
        dirpath = train_free_v2_dir / dirname
        if dirpath.exists():
            print(f"✓ {dirname}/")
        else:
            print(f"✗ {dirname}/ (MISSING)")
            all_ok = False
    
    return all_ok


def run_tests():
    """Run unit tests."""
    print("\n" + "=" * 80)
    print("Running unit tests...")
    print("=" * 80)
    
    test_dir = Path(__file__).parent / 'tests'
    test_file = test_dir / 'test_sinkhorn_loss.py'
    
    if not test_file.exists():
        print("✗ Test file not found")
        return False
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(test_file)],
            timeout=300  # 5 minute timeout
        )
        return result.returncode == 0
    except Exception as e:
        print(f"✗ Test execution failed: {e}")
        return False


def print_quickstart():
    """Print quick-start instructions."""
    print("\n" + "=" * 80)
    print("Quick Start Guide")
    print("=" * 80)
    
    print("""
1. Basic usage:
   python train_free_v2/sample_dps.py \\
       --image /path/to/target.png \\
       --base_ckpt config/GBN/model.ckpt \\
       --config config/GBN/config.json

2. With custom parameters:
   python train_free_v2/sample_dps.py \\
       --image /path/to/target.png \\
       --lambda_scale 1.5 \\
       --timesteps 500 \\
       --grad_clip 1.0

3. Batch generation:
   python train_free_v2/sample_dps.py \\
       --image /path/to/target.png \\
       --n_samples 5

4. Run tests:
   python -m pytest train_free_v2/tests/ -v
   # or
   python train_free_v2/tests/test_sinkhorn_loss.py

5. See full documentation:
   cat train_free_v2/README.md

For more help:
   python train_free_v2/sample_dps.py --help
""")


def main():
    """Run all checks."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " train_free_v2 Setup & Validation ".center(78) + "║")
    print("╚" + "=" * 78 + "╝")
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Directory Structure", check_structure),
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"\n✗ {name} check failed: {e}")
            results[name] = False
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    all_passed = all(results.values())
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    # Run tests only if setup passed
    if all_passed:
        print("\n" + "=" * 80)
        print("Optional: Running unit tests...")
        print("=" * 80)
        try:
            test_ok = run_tests()
            print(f"{'✓ PASS' if test_ok else '✗ FAIL'}: Unit Tests")
        except Exception as e:
            print(f"✗ FAIL: Unit Tests ({e})")
    
    # Quick-start
    print_quickstart()
    
    # Final status
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ train_free_v2 is ready to use!")
    else:
        print("✗ Please fix the issues above before using train_free_v2")
    print("=" * 80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
