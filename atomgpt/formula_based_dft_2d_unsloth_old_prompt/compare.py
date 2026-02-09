import json
import csv
from collections import Counter
from fractions import Fraction
from math import gcd as _gcd
from functools import reduce
import numpy as np

# ---------------- helpers ----------------
def gcd(a, b):  # keep your name
    while b:
        a, b = b, a % b
    return a

def _gcd_list(ints):
    ints = [abs(int(x)) for x in ints if int(x) != 0]
    if not ints:
        return 1
    return reduce(_gcd, ints)

def reduce_formula(elements):
    """
    elements: list like ['Al','H','H','O','O','O','O'].
    Returns dict of reduced integer ratios, e.g. {'Al':1,'H':2,'O':4}.
    """
    cnt = Counter(elements)
    vals = list(cnt.values())
    g = _gcd_list(vals) if vals else 1
    return {k: Fraction(v, g) for k, v in cnt.items()}

def formula_to_string(red: dict):
    """
    Convert reduced dict (Fractions) to a formula string.
    Uses Hill-like order: C, H, then alphabetical.
    Fractions should be integers here; if not, shows as a/b.
    """
    # Promote to integers where possible
    def coeff_str(fr):
        if fr.denominator == 1:
            n = fr.numerator
            return "" if n == 1 else str(n)
        return f"{fr.numerator}/{fr.denominator}"

    keys = list(red.keys())
    # Hill-ish ordering
    if 'C' in keys:
        ordered = ['C'] + (['H'] if 'H' in keys else []) + sorted([k for k in keys if k not in ('C','H')])
    else:
        ordered = sorted(keys)
    parts = [f"{el}{coeff_str(red[el])}" for el in ordered]
    return "".join(parts) if parts else ""

def compare_structures(target, predicted):
    # stoichiometry
    tgt_red = reduce_formula(target["elements"])
    pred_red = reduce_formula(predicted["elements"])
    stoichiometry_match = tgt_red == pred_red

    # lattice and angles
    t_abc = np.array(target["abc"], dtype=float)
    p_abc = np.array(predicted["abc"], dtype=float)
    t_ang = np.array(target["angles"], dtype=float)
    p_ang = np.array(predicted["angles"], dtype=float)

    abs_abc = np.abs(t_abc - p_abc)
    abs_ang = np.abs(t_ang - p_ang)

    return {
        "stoichiometry_match": stoichiometry_match,
        "target_formula": formula_to_string(tgt_red),
        "predicted_formula": formula_to_string(pred_red),
        "a_gt": t_abc[0], "b_gt": t_abc[1], "c_gt": t_abc[2],
        "alpha_gt": t_ang[0], "beta_gt": t_ang[1], "gamma_gt": t_ang[2],
        "a_pred": p_abc[0], "b_pred": p_abc[1], "c_pred": p_abc[2],
        "alpha_pred": p_ang[0], "beta_pred": p_ang[1], "gamma_pred": p_ang[2],
        "a_mae": abs_abc[0], "b_mae": abs_abc[1], "c_mae": abs_abc[2],
        "alpha_mae": abs_ang[0], "beta_mae": abs_ang[1], "gamma_mae": abs_ang[2],
    }

def process_data(data):
    rows = []
    for item in data:
        c = compare_structures(item["target"], item["predicted"])
        rows.append({
            "id": item["id"],
            "stoichiometry_match": c["stoichiometry_match"],
            "target_formula": c["target_formula"],
            "predicted_formula": c["predicted_formula"],
            "a_gt": c["a_gt"], "b_gt": c["b_gt"], "c_gt": c["c_gt"],
            "alpha_gt": c["alpha_gt"], "beta_gt": c["beta_gt"], "gamma_gt": c["gamma_gt"],
            "a_pred": c["a_pred"], "b_pred": c["b_pred"], "c_pred": c["c_pred"],
            "alpha_pred": c["alpha_pred"], "beta_pred": c["beta_pred"], "gamma_pred": c["gamma_pred"],
            "a_mae": c["a_mae"], "b_mae": c["b_mae"], "c_mae": c["c_mae"],
            "alpha_mae": c["alpha_mae"], "beta_mae": c["beta_mae"], "gamma_mae": c["gamma_mae"],
        })
    return rows

def load_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
    return None

def write_to_csv(results, output_file):
    fieldnames = [
        "id", "stoichiometry_match", "target_formula", "predicted_formula",
        "a_gt","b_gt","c_gt","alpha_gt","beta_gt","gamma_gt",
        "a_pred","b_pred","c_pred","alpha_pred","beta_pred","gamma_pred",
        "a_mae","b_mae","c_mae","alpha_mae","beta_mae","gamma_mae"
    ]
    with open(output_file, 'w', newline='') as csvfile:
        w = csv.DictWriter(csvfile, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

def main():
    file_path = "/projects/p32726/microscopy-gpt/atomgpt/atomgpt/formula_based_dft_2d_unsloth_old_prompt/gemma-3-27b-it-unsloth-bnb-4bit/gemma3-27B.json"
    data = load_json_file(file_path)
    if data is None:
        return
    results = process_data(data)
    for r in results[:10]:  # print a few; remove slice to print all
        print(f"ID: {r['id']}")
        print(f"Stoichiometry match: {r['stoichiometry_match']}")
        print(f"Target formula: {r['target_formula']}  |  Predicted formula: {r['predicted_formula']}")
        print(f"a_gt={r['a_gt']}, b_gt={r['b_gt']}, c_gt={r['c_gt']}, "
              f"alpha_gt={r['alpha_gt']}, beta_gt={r['beta_gt']}, gamma_gt={r['gamma_gt']}")
        print(f"a_pred={r['a_pred']}, b_pred={r['b_pred']}, c_pred={r['c_pred']}, "
              f"alpha_pred={r['alpha_pred']}, beta_pred={r['beta_pred']}, gamma_pred={r['gamma_pred']}")
        print(f"a_mae={r['a_mae']}, b_mae={r['b_mae']}, c_mae={r['c_mae']}, "
              f"alpha_mae={r['alpha_mae']}, beta_mae={r['beta_mae']}, gamma_mae={r['gamma_mae']}")
        print("-"*60)
    output_file = "error_with_details.csv"
    write_to_csv(results, output_file)
    print(f"Results written to {output_file}")

if __name__ == "__main__":
    main()
