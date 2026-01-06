from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from tqdm import tqdm

from faultloc.pipeline.pipeline import run_faultloc_for_instance

console = Console()

def load_dataset(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported dataset JSON structure. Expected a list or {data:[...] }")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=False, help="Path to swebench_lite.json")
    ap.add_argument("--repos-root", type=str, required=False, help="Root directory containing <instance_id>/ repos")
    ap.add_argument("--artifacts", type=str, required=False, help="Output artifacts directory")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--max-bugs", type=int, default=0, help="0 means all")
    ap.add_argument("--evaluate", action="store_true", help="Evaluate using patch field if present")
    args = ap.parse_args()
    
    dataset_path = Path("-")
    repos_root = Path("-")
    artifacts_root = Path("-")
    artifacts_root.mkdir(parents=True, exist_ok=True)
    args.max_bugs = 1

    ds = load_dataset(dataset_path)
    if args.max_bugs and args.max_bugs > 0:
        ds = ds[: args.max_bugs]

    results = []
    for item in tqdm(ds, desc="FaultLoc"):
        iid = item.get("instance_id") or item.get("id")
        ps = item.get("problem_statement") or item.get("problem statement")
        if not iid or not ps:
            continue
        repo_path = repos_root / str(iid)
        if not repo_path.exists():
            console.print(f"[yellow]Repo folder not found for {iid}: {repo_path}[/yellow]")
            continue
        patch = item.get("patch") if args.evaluate else None
        try:
            out = run_faultloc_for_instance(
                instance_id=str(iid),
                problem_statement=str(ps),
                repo_path=repo_path,
                artifacts_root=artifacts_root,
                topk=int(args.topk),
                evaluate_patch=str(patch) if patch else None,
            )
            results.append(out)
        except Exception as e:
            console.print(f"[red]Failed {iid}: {e}[/red]")
            results.append({"instance_id": str(iid), "error": str(e)})

    out_path = artifacts_root / "output.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]Done. Summary written to {out_path}[/green]")

if __name__ == "__main__":
    main()
