#!/usr/bin/env python3
"""Copy ~/ramses-development *.md into Obsidian vault with frontmatter and link map."""

from __future__ import annotations

import json
import re
import shutil
from datetime import date
from pathlib import Path

REPO_ROOT = Path("/Users/moseley/ramses-development")
VAULT = Path("/Users/moseley/Documents/Obsidian Vault")
TODAY = date.today().isoformat()

SKIP_PARTS = (
    "/.git/",
    "/.pytest_cache/",
    "/site-packages/",
    "/.venv",
    "/.venv_icgen/",
    "/graficICgen/.venv/",
)

# source relative path -> vault folder (under VAULT)
ROUTE_PREFIXES: list[tuple[str, str, str]] = [
    # (prefix, vault_subdir, project tag)
    ("docs/adr/", "40-ADRs", "mini-ramses"),
    ("docs/triage/", "20-projects/mini-ramses/triage", "mini-ramses"),
    ("docs/", "20-projects/mini-ramses/refs", "mini-ramses"),
    ("artifacts/gpu-pm/", "20-projects/mini-ramses/artifacts/gpu-pm", "mini-ramses"),
    ("gpu-mhd-archive/", "20-projects/mini-ramses/archive/gpu-mhd", "mini-ramses"),
    ("old_gpu_part_md/", "20-projects/mini-ramses/archive/old-gpu-part", "mini-ramses"),
    ("mini-ramses-dev/doc/", "20-projects/mini-ramses/namelist-docs", "mini-ramses"),
    ("mini-ramses-dev/gpu/", "20-projects/mini-ramses/notes", "mini-ramses"),
    ("mini-ramses-dev/", "20-projects/mini-ramses/notes", "mini-ramses"),
    ("mini-ramses-dev-2/doc/", "20-projects/tracers/namelist-docs", "tracers"),
    ("mini-ramses-dev-2/gpu/", "20-projects/tracers/notes", "tracers"),
    ("mini-ramses-dev-2/", "20-projects/tracers/notes", "tracers"),
    ("mini-ramses/doc/", "20-projects/mini-ramses/legacy/doc", "mini-ramses"),
    ("mini-ramses/", "20-projects/mini-ramses/legacy", "mini-ramses"),
    ("ramses/doc/wiki/", "20-projects/ramses/wiki", "ramses"),
    ("ramses/doc/dev_docs/", "20-projects/ramses/dev-docs", "ramses"),
    ("ramses/doc/", "20-projects/ramses/docs", "ramses"),
    ("ramses/tests/", "20-projects/ramses/tests", "ramses"),
    ("ramses/", "20-projects/ramses/repo", "ramses"),
    ("ramses-pic/doc/wiki/", "20-projects/ramses/ramses-pic/wiki", "ramses"),
    ("ramses-pic/doc/dev_docs/", "20-projects/ramses/ramses-pic/dev-docs", "ramses"),
    ("ramses-pic/", "20-projects/ramses/ramses-pic", "ramses"),
    ("rundir/", "20-projects/mini-ramses/rundir", "mini-ramses"),
    ("tests/", "20-projects/tracers/tests", "tracers"),
    ("gpu_part_diff_harness/", "20-projects/mini-ramses/artifacts/gpu-part-diff", "mini-ramses"),
    ("particle_analysis/", "20-projects/mini-ramses/artifacts/particle-analysis", "mini-ramses"),
    ("graficICgen/", "20-projects/mini-ramses/tools/graficICgen", "mini-ramses"),
]

ROOT_PLAN_NAMES = {
    "mhd_plan.md": "mini-ramses — mhd plan",
    "mpi_gpu_plan.md": "mini-ramses — mpi gpu plan",
    "MHD_PLAN_STATE.md": "mini-ramses — MHD plan state",
    "GPU_PM_STATE.md": "mini-ramses — GPU PM state",
    "WORKFLOW.md": "mini-ramses — GPU PM workflow",
    "nsubgrid_mhd_fix.md": "mini-ramses — nsubgrid MHD fix",
    "nsubgrid_amr_plan.md": "mini-ramses — nsubgrid AMR plan",
    "kick_coop_gather_notes.md": "mini-ramses — kick coop gather notes",
    "kick_gather_warp_coop_plan.md": "mini-ramses — kick gather warp coop plan",
    "gpu_kick_drift_part_plan.md": "mini-ramses — gpu kick drift part plan",
    "part_optimization_goals.md": "mini-ramses — part optimization goals",
    "simplifying_gpu_part.md": "mini-ramses — simplifying gpu part",
    "leaning_out_part_arrays.md": "mini-ramses — leaning out part arrays",
    "planned_next_stage_edits.md": "mini-ramses — planned next stage edits",
    "overhaul.md": "mini-ramses — overhaul",
    "athenak_vs_mini-ramses_comparison.md": "mini-ramses — athenak comparison",
    "day1_meeting_notes.md": "mini-ramses — day1 meeting notes",
    "warpx_improvements.md": "mini-ramses — warpx improvements",
    "3_cic_edits.md": "mini-ramses — 3 cic edits",
    "to_do.md": "mini-ramses — to do",
}

PROMPT_ROOT = re.compile(r".*_prompt\.md$|.*prompt.*\.md$", re.I)


def should_skip(path: Path) -> bool:
    s = str(path)
    return any(p in s for p in SKIP_PARTS)


def route(rel: str) -> tuple[str, str]:
    """Return (vault_subdir, project)."""
    name = Path(rel).name
    if rel.count("/") == 0:
        if PROMPT_ROOT.match(name):
            return "20-projects/mini-ramses/prompts", "mini-ramses"
        if name in ROOT_PLAN_NAMES or name.endswith(".md"):
            return "20-projects/mini-ramses/plans", "mini-ramses"
    for prefix, subdir, project in ROUTE_PREFIXES:
        if rel.startswith(prefix) or rel == prefix.rstrip("/"):
            return subdir, project
    return "20-projects/mini-ramses/misc", "mini-ramses"


def vault_basename(rel: str, src_name: str) -> str:
    rel_posix = rel.replace("\\", "/")
    if rel_posix.count("/") == 0 and src_name in ROOT_PLAN_NAMES:
        return ROOT_PLAN_NAMES[src_name] + ".md"
    stem = Path(src_name).stem
    if stem.upper() == "README":
        parent = Path(rel_posix).parent.name
        if parent and parent != ".":
            return f"{parent} — README.md"
        return "README — " + Path(rel_posix).parts[0] + ".md"
    if "/doc/wiki/" in rel_posix or rel_posix.startswith("ramses/doc/wiki/") or "ramses-pic/doc/wiki/" in rel_posix:
        return f"RAMSES wiki — {stem}.md"
    if "/namelist-docs/" in rel_posix or rel_posix.endswith("_params.md") or stem.endswith("_params"):
        label = stem.replace("_", " ")
        return f"namelist — {label}.md"
    if rel_posix.startswith("docs/adr/"):
        if stem.startswith("0001-"):
            return "ADR-0003 — GPU PM documentation restructure.md"
        return stem.replace("-", " — ", 1) if stem[0].isdigit() else f"ADR — {stem}.md"
    if "/triage/" in rel_posix:
        return f"triage — {stem}.md"
    if "/prompts/" in rel_posix or PROMPT_ROOT.match(src_name):
        return f"prompt — {stem}.md"
    if "/plans/" in rel_posix or rel_posix.count("/") == 0:
        return f"mini-ramses — {stem.replace('_', ' ')}.md"
    return stem.replace("_", " ") + ".md"


def wikilink_name(vault_filename: str) -> str:
    return Path(vault_filename).stem


def rewrite_md_links(body: str, link_map: dict[str, str]) -> str:
    """Rewrite relative .md links to Obsidian wikilinks where mapped."""

    def repl_md(m: re.Match) -> str:
        text, target = m.group(1), m.group(2)
        key = target.split("#")[0]
        key = key.lstrip("./")
        while key.startswith("../"):
            key = key[3:]
        resolved = link_map.get(key) or link_map.get(Path(key).name)
        if resolved:
            anchor = ""
            if "#" in target:
                anchor = target[target.index("#") :]
            return f"[[{resolved}{anchor}|{text}]]" if text != resolved else f"[[{resolved}{anchor}]]"
        return m.group(0)

    body = re.sub(r"\[([^\]]*)\]\(([^)]+\.md[^)]*)\)", repl_md, body)
    return body


def frontmatter(project: str, rel: str, note_type: str) -> str:
    tags = ["type/repo-doc", f"project/{project}"]
    if "adr" in rel or note_type == "adr":
        tags = ["type/adr", f"project/{project}"]
    elif "triage" in rel:
        tags = ["type/triage", f"project/{project}"]
    elif "wiki" in rel:
        tags = ["type/wiki", f"project/{project}"]
    elif "_params" in rel or "namelist-docs" in rel:
        tags = ["type/namelist", f"project/{project}"]
    elif "prompt" in rel:
        tags = ["type/prompt", f"project/{project}"]
    tag_lines = "\n".join(f"  - {t}" for t in tags)
    return (
        f"---\n"
        f"tags:\n{tag_lines}\n"
        f"created: {TODAY}\n"
        f"source: ~/ramses-development/{rel}\n"
        f"---\n\n"
        f"> Relocated from `~/ramses-development/{rel}`. Hub: [[mini-ramses — hub]] · [[ramses — hub]] · [[tracers — hub]]\n\n"
    )


def main() -> None:
    manifest: list[dict] = []
    link_map: dict[str, str] = {}
    copies: list[tuple[Path, Path, str, str]] = []

    for src in sorted(REPO_ROOT.rglob("*")):
        if src.suffix.lower() != ".md" or not src.is_file():
            continue
        if should_skip(src):
            continue
        rel = str(src.relative_to(REPO_ROOT)).replace("\\", "/")
        subdir, project = route(rel)
        vname = vault_basename(rel, src.name)
        dest_dir = VAULT / subdir
        dest = dest_dir / vname
        copies.append((src, dest, rel, project))
        link_map[rel] = wikilink_name(vname)
        link_map[src.name] = wikilink_name(vname)
        # also map without extension paths used in links
        link_map[rel.rsplit(".", 1)[0]] = wikilink_name(vname)

    for src, dest, rel, project in copies:
        dest.parent.mkdir(parents=True, exist_ok=True)
        body = src.read_text(encoding="utf-8", errors="replace")
        if body.startswith("---\n"):
            body = re.sub(r"^---\n.*?\n---\n", "", body, count=1, flags=re.S)
        body = rewrite_md_links(body, link_map)
        note_type = "adr" if "docs/adr" in rel else "doc"
        out = frontmatter(project, rel, note_type) + body
        dest.write_text(out, encoding="utf-8")
        manifest.append(
            {
                "source": rel,
                "vault_path": str(dest.relative_to(VAULT)),
                "wikilink": wikilink_name(dest.name),
                "project": project,
            }
        )

    # Index MOC
    by_project: dict[str, list[dict]] = {}
    for item in manifest:
        by_project.setdefault(item["project"], []).append(item)

    moc_body = f"""---
tags:
  - type/moc
  - project/mini-ramses
created: {TODAY}
aliases:
  - ramses-development docs index
  - repo docs MOC
---

# ramses-development — repo docs MOC

Index of markdown relocated from `~/ramses-development/` into this vault ({len(manifest)} notes). Regenerate with `.cursor/scripts/sync_md_to_obsidian.py`.

## Hubs

- [[mini-ramses — hub]] · [[ramses — hub]] · [[tracers — hub]]
- [[GPU optimization MOC]] · [[Validation & test problems MOC]]

## mini-ramses ({len(by_project.get('mini-ramses', []))})

### Plans & state
"""
    key_plans = [
        "mini-ramses — mpi gpu plan",
        "mini-ramses — mhd plan",
        "mini-ramses — GPU PM state",
        "mini-ramses — GPU PM workflow",
        "mini-ramses — MHD plan state",
        "mini-ramses — nsubgrid MHD fix",
        "mini-ramses — kick coop gather notes",
    ]
    for k in key_plans:
        moc_body += f"- [[{k}]]\n"

    moc_body += "\n### Namelist reference (mini-ramses-dev)\n"
    for item in sorted(by_project.get("mini-ramses", []), key=lambda x: x["vault_path"]):
        if "namelist-docs" in item["vault_path"]:
            moc_body += f"- [[{item['wikilink']}]]\n"

    moc_body += f"\n## ramses ({len(by_project.get('ramses', []))})\n\n- [[RAMSES wiki — Home]] · [[RAMSES wiki — Tracers]] · [[RAMSES wiki — Runtime Parameters]]\n"
    moc_body += f"\n## tracers ({len(by_project.get('tracers', []))})\n"
    for item in sorted(by_project.get("tracers", []), key=lambda x: x["vault_path"]):
        if "namelist-docs" in item["vault_path"] or "gc_term" in item["source"]:
            moc_body += f"- [[{item['wikilink']}]]\n"

    moc_path = VAULT / "MOCs/ramses-development — repo docs MOC.md"
    moc_path.write_text(moc_body, encoding="utf-8")

    manifest_path = VAULT / "20-projects/mini-ramses/_sync_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Copied {len(manifest)} notes to vault")
    print(f"MOC: {moc_path}")


if __name__ == "__main__":
    main()
