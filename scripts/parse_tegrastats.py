#!/usr/bin/env python3
"""Estrae metriche aggregate da un file di log tegrastats (Jetson TX2 / L4T).

Formato riga tegrastats (esempio TX2, L4T r32.7):
  06-30-2026 10:15:01 RAM 3456/7829MB (lfb 1x2MB) SWAP 0/3914MB (cached 0MB)
  CPU [12%@2035,11%@2035,...] EMC_FREQ 50% GR3D_FREQ 87% PLL@42C CPU@48C ...
  VDD_IN 5234mW VDD_CPU 987mW VDD_GPU 1234mW ...

Usage:
  python3 scripts/parse_tegrastats.py tegra_standard.txt
  python3 scripts/parse_tegrastats.py tegra_standard.txt --json  (default)
  python3 scripts/parse_tegrastats.py tegra_standard.txt --pretty
"""
import argparse
import json
import re
import sys
from pathlib import Path


def _extract(line: str):
    row = {}

    # GR3D_FREQ (GPU utilization %)
    m = re.search(r"GR3D_FREQ\s+(\d+)%", line)
    if m:
        row["gr3d_pct"] = int(m.group(1))

    # CPU [12%@2035,14%@2035,...] — media delle core
    m = re.search(r"CPU\s+\[([^\]]+)\]", line)
    if m:
        core_pcts = [int(x.split("%")[0]) for x in m.group(1).split(",") if "%" in x]
        if core_pcts:
            row["cpu_pct_mean"] = sum(core_pcts) / len(core_pcts)

    # RAM XXXX/YYYYMB
    m = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
    if m:
        row["ram_used_mb"] = int(m.group(1))
        row["ram_total_mb"] = int(m.group(2))

    # VDD_IN — due formati:
    #   TX2/L4T r32: "VDD_IN 4290/4290" (cur/avg mW, no unità)
    #   altre piattaforme: "VDD_IN 4290mW"
    m = re.search(r"VDD_IN\s+(\d+)/(\d+)", line)
    if m:
        row["vdd_in_mw"] = int(m.group(1))  # valore corrente
    else:
        m = re.search(r"VDD_IN\s+(\d+)mW", line)
        if m:
            row["vdd_in_mw"] = int(m.group(1))

    # POM_5V_IN (alternativo su alcuni TX2)
    m = re.search(r"POM_5V_IN\s+(\d+)/(\d+)", line)
    if m:
        row["pom_5v_mw"] = int(m.group(1))

    # Temperature — TX2: GPU@42C BCPU@43C MCPU@43C PLL@42C thermal@42.6C ...
    for sensor in ("GPU", "BCPU", "MCPU", "CPU", "AO", "PMIC", "thermal", "PLL"):
        m = re.search(rf"{sensor}@(\d+\.?\d*)C", line)
        if m:
            row[f"temp_{sensor.lower()}_c"] = float(m.group(1))

    return row if row else None


def parse_file(path: str) -> dict:
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            r = _extract(line.strip())
            if r:
                rows.append(r)

    if not rows:
        return {"error": "nessuna riga parsata", "n_samples": 0, "source": path}

    def col(key):
        return [r[key] for r in rows if key in r]

    def avg(vals):
        return round(sum(vals) / len(vals), 1) if vals else None

    def peak(vals):
        return round(max(vals), 1) if vals else None

    gr3d = col("gr3d_pct")
    cpu  = col("cpu_pct_mean")
    vdd  = col("vdd_in_mw")
    pom  = col("pom_5v_mw")
    ram  = col("ram_used_mb")
    power_vals = vdd or pom

    result = {
        "source": str(path),
        "n_samples": len(rows),
        "gpu_gr3d_pct_mean": avg(gr3d),
        "gpu_gr3d_pct_peak": peak(gr3d),
        "cpu_pct_mean": avg(cpu),
        "cpu_pct_peak": peak(cpu),
        "ram_used_mb_peak": peak(ram),
        "power_mw_mean": avg(power_vals),
        "power_mw_peak": peak(power_vals),
        "power_source": "VDD_IN" if vdd else ("POM_5V_IN" if pom else None),
    }

    # Temperature per sensore (TX2: bcpu/mcpu sono i sensori CPU principali)
    for sensor in ("gpu", "bcpu", "mcpu", "cpu", "ao", "pmic", "thermal", "pll"):
        vals = col(f"temp_{sensor}_c")
        if vals:
            result[f"temp_{sensor}_c_peak"] = peak(vals)
            result[f"temp_{sensor}_c_mean"] = avg(vals)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Parser log tegrastats → JSON metriche")
    p.add_argument("logfile", help="File log tegrastats")
    p.add_argument("--pretty", action="store_true", help="Output human-readable invece di JSON puro")
    args = p.parse_args()

    result = parse_file(args.logfile)

    if args.pretty:
        print(f"Sorgente  : {result['source']}")
        print(f"Campioni  : {result['n_samples']}")
        print(f"GPU GR3D  : medio={result.get('gpu_gr3d_pct_mean')}%  picco={result.get('gpu_gr3d_pct_peak')}%")
        print(f"CPU       : medio={result.get('cpu_pct_mean')}%  picco={result.get('cpu_pct_peak')}%")
        print(f"RAM       : picco={result.get('ram_used_mb_peak')} MB")
        print(f"Potenza   : medio={result.get('power_mw_mean')} mW  picco={result.get('power_mw_peak')} mW  ({result.get('power_source')})")
        for sensor in ("cpu", "gpu", "ao"):
            if f"temp_{sensor}_c_peak" in result:
                print(f"Temp {sensor.upper():6}: media={result[f'temp_{sensor}_c_mean']}°C  picco={result[f'temp_{sensor}_c_peak']}°C")
    else:
        print(json.dumps(result, indent=2))
