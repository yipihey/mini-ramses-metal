#!/usr/bin/env python3
"""
Generate bin/deps.mk — Fortran module dependency rules for parallel make.

Run from the repository root:
    python3 utils/py/gen_deps.py > bin/deps.mk

The output is included by bin/Makefile via:
    -include deps.mk

Dependencies involving GPU objects (.cuf sources) are wrapped in
ifeq ($(COMPILER),NVHPC) so they are invisible to GNU/Intel builds.
Dependencies inside #ifdef physics blocks (RT, RTZ, HYDRO, ...) are
wrapped in the corresponding ifeq ($(VAR),1) Makefile guard.
"""

import os
import re
import sys
from collections import defaultdict

SRCDIRS = [
    'amr', 'hydro', 'pm', 'poisson', 'cooling', 'rt', 'gpu',
    'turb', 'clump', 'sink', 'star', 'feedback', 'mdl1', 'lightcone',
]
EXTS = ('.f90', '.cuf')

BASENAME_OVERRIDE = {
    'cub_inclusive_scan':    'cub_inclusive_scan_f',
    'cub_module_radix_sort': 'cub_module_radix_sort_f',
}

# Fortran preprocessor macro -> Makefile variable
MACRO_TO_MAKE = {
    'HYDRO': 'HYDRO',
    'MHD':   'MHD',
    'GRAV':  'GRAV',
    'RT':    'RT',
    'RTZ':   'RTZ',
    'TURB':  'TURB',
}


def parse_uses(path):
    """Parse a Fortran source file and return [(module_name, guard), ...].

    guard is a Makefile ifeq string (e.g. 'ifeq ($(RTZ),1)') or None for
    unconditional USE statements.  Only the innermost recognised #ifdef
    macro contributes to the guard; unknown macros are ignored.
    """
    uses, stack = [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            m = re.match(r'#\s*ifdef\s+(\w+)', s)
            if m:
                stack.append((m.group(1).upper(), False))
                continue
            m = re.match(r'#\s*ifndef\s+(\w+)', s)
            if m:
                stack.append((m.group(1).upper(), True))
                continue
            m = re.match(r'#\s*if\s+defined\s*\(\s*(\w+)\s*\)', s)
            if m:
                stack.append((m.group(1).upper(), False))
                continue
            if re.match(r'#\s*else\b', s) and stack:
                macro, neg = stack[-1]
                stack[-1] = (macro, not neg)
                continue
            if re.match(r'#\s*endif\b', s) and stack:
                stack.pop()
                continue
            m = re.match(r'\s*use\s+(\w+)', s, re.IGNORECASE)
            if m:
                guard = None
                for macro, neg in reversed(stack):
                    if macro in MACRO_TO_MAKE:
                        val = '0' if neg else '1'
                        guard = f'ifeq ($({MACRO_TO_MAKE[macro]}),{val})'
                        break
                uses.append((m.group(1).lower(), guard))
    return uses


def main():
    # Pass 1: map module name -> object name, collect GPU-only objects
    mod_to_obj = {}
    gpu_only = set()  # objects that come from .cuf files

    for d in SRCDIRS:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not any(fname.endswith(e) for e in EXTS):
                continue
            base = re.sub(r'\.(f90|cuf)$', '', fname)
            obj = BASENAME_OVERRIDE.get(base, base)
            if fname.endswith('.cuf'):
                gpu_only.add(obj)
            with open(os.path.join(d, fname)) as f:
                for line in f:
                    m = re.match(r'^\s*module\s+(\w+)\s*$', line, re.IGNORECASE)
                    if m:
                        mod_to_obj[m.group(1).lower()] = obj

    # C wrapper objects (compiled by nvcc, not nvfortran, also NVHPC-only)
    gpu_only.update(['cub_inclusive_scan_c', 'cub_module_radix_sort_c'])

    # Pass 2: collect (obj, guard) -> set of dep objects
    deps = defaultdict(set)
    for d in SRCDIRS:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not any(fname.endswith(e) for e in EXTS):
                continue
            base = re.sub(r'\.(f90|cuf)$', '', fname)
            obj = BASENAME_OVERRIDE.get(base, base)
            for mod, src_guard in parse_uses(os.path.join(d, fname)):
                if mod not in mod_to_obj or mod_to_obj[mod] == obj:
                    continue
                dep_obj = mod_to_obj[mod]
                # GPU objects override other guards
                if obj in gpu_only or dep_obj in gpu_only:
                    guard = 'ifeq ($(COMPILER),NVHPC)'
                else:
                    guard = src_guard
                deps[(obj, guard)].add(dep_obj)

    # Emit Makefile rules grouped by guard
    print("# Auto-generated Fortran module dependency rules for parallel make.")
    print("# Regenerate: python3 utils/py/gen_deps.py > bin/deps.mk")
    print()

    all_guards = sorted(
        set(g for (_, g) in deps),
        key=lambda x: (1, x) if x else (0, ''),
    )
    for guard in all_guards:
        block = {
            o: sorted(ds)
            for (o, g), ds in sorted(
                deps.items(), key=lambda kv: (kv[0][0], kv[0][1] or '')
            )
            if g == guard
        }
        if not block:
            continue
        if guard:
            print(guard)
        for obj in sorted(block):
            dep_list = ' '.join(d + '.o' for d in block[obj])
            print(f"{obj}.o: {dep_list}")
        if guard:
            print("endif")
        print()


if __name__ == '__main__':
    # Allow running from bin/ or repo root
    if os.path.isdir('bin') and not os.path.isdir('amr'):
        os.chdir('..')
    elif not os.path.isdir('amr'):
        print("Error: run from the repository root or bin/", file=sys.stderr)
        sys.exit(1)
    main()
