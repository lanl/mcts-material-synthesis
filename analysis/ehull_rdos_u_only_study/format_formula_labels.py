#!/usr/bin/env python3
"""
Format compound formulas to {RE}{TM}_{6}{GroupIV}_{6} format.
Properly identifies which element is which regardless of input order.
"""

import sys

# Element definitions
RARE_EARTHS = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'U']
TRANSITION_METALS = ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                     'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                     'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']
GROUP_IV = ['Si', 'Ge', 'Sn', 'Pb']

def parse_formula(formula):
    """
    Parse formula to extract RE, TM, and Group IV elements.
    Handles various input formats by identifying element types.
    """
    import re

    # Extract all element-number pairs
    # Pattern: element (1-2 letters) followed by optional number
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)

    re_elem = None
    tm_elem = None
    giv_elem = None

    for elem, count in matches:
        if not elem:  # Skip empty matches
            continue

        if elem in RARE_EARTHS:
            re_elem = elem
        elif elem in TRANSITION_METALS:
            tm_elem = elem
        elif elem in GROUP_IV:
            giv_elem = elem

    if re_elem and tm_elem and giv_elem:
        return f"{re_elem}{tm_elem}_{{6}}{giv_elem}_{{6}}"
    else:
        # Fallback: return original formula if we can't parse it
        return formula

if __name__ == '__main__':
    if len(sys.argv) > 1:
        formula = sys.argv[1]
        print(parse_formula(formula))
    else:
        # Test cases
        test_cases = [
            'Ir6LuSi6',
            'GdIr6Si6',
            'Ge6TmV6',
            'LuIr6Si6',
            'Ge6YbZr6'
        ]

        for formula in test_cases:
            formatted = parse_formula(formula)
            print(f"{formula:15s} -> {formatted}")
