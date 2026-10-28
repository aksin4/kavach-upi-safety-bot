"""Run test_set.csv against the running server and log results to the test_results table.

Usage:  uvicorn app:app --port 8000   (in another terminal)
        python run_tests.py
A test passes if the reply contains ANY of the pipe-separated expected keywords (case-insensitive).
This is a quick automated check; review failures manually, since keyword matching is imperfect.
"""
import csv
import sqlite3
import sys
import time
import uuid
from contextlib import closing

import requests

URL = "http://127.0.0.1:8000/api/chat"

with closing(sqlite3.connect("chat.db")) as con, con:
    con.execute(
        "CREATE TABLE IF NOT EXISTS test_results(test_id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT, "
        "expected TEXT, actual TEXT, correct INTEGER, latency_ms INTEGER)"
    )

rows = list(csv.DictReader(open("test_set.csv", encoding="utf-8")))
passed, by_type, out = 0, {}, []
for r in rows:
    sid = str(uuid.uuid4())
    t0 = time.perf_counter()
    try:
        resp = requests.post(URL, json={"session_id": sid, "message": r["query"]}, timeout=40)
        reply = resp.json().get("reply") or resp.json().get("detail", "")
    except Exception as e:  # noqa: BLE001
        reply = f"ERROR: {e}"
    ms = int((time.perf_counter() - t0) * 1000)
    ok = any(k.strip().lower() in reply.lower() for k in r["expected_keywords"].split("|") if k.strip())
    passed += ok
    by_type.setdefault(r["type"], []).append(ok)
    with closing(sqlite3.connect("chat.db")) as con, con:
        con.execute(
            "INSERT INTO test_results(query, expected, actual, correct, latency_ms) VALUES(?,?,?,?,?)",
            (r["query"], r["expected_keywords"], reply, int(ok), ms),
        )
    out.append({**r, "actual": reply, "correct": int(ok), "latency_ms": ms})
    print(("PASS" if ok else "FAIL"), f"{ms:>5} ms |", r["type"], "|", r["query"][:60])
    time.sleep(0.5)

with open("results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)

print(f"\n{passed}/{len(rows)} passed")
for k, v in by_type.items():
    print(f"  {k:<14} {sum(v)}/{len(v)}")
sys.exit(0 if passed == len(rows) else 1)
