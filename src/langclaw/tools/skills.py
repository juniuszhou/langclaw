"""Skill loader: load SKILL.md folders and inject instructions into context."""

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class Skill(BaseModel):
    """A skill: name, description, and instruction content from SKILL.md."""

    name: str
    description: str = ""
    path: Path
    content: str = ""


def load_skill(skill_dir: Path) -> Optional[Skill]:
    """Load a single skill from a directory containing SKILL.md."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None
    try:
        content = skill_md.read_text(encoding="utf-8", errors="replace")
        name = skill_dir.name
        # First line or first paragraph as description
        lines = content.strip().split("\n")
        desc = lines[0].strip() if lines else ""
        if desc.startswith("#"):
            desc = desc.lstrip("#").strip()
        return Skill(name=name, description=desc, path=skill_dir, content=content)
    except Exception:
        return None


def load_skills(skills_root: Path, names: Optional[List[str]] = None) -> List[Skill]:
    """Load skills from a root directory. Each subdir with SKILL.md is a skill.

    Args:
        skills_root: Root directory containing skill subdirectories.
        names: Optional list of skill names to load. If None, load all.

    Returns:
        List of Skill objects.
    """
    if not skills_root.exists() or not skills_root.is_dir():
        return []
    skills = []
    for subdir in skills_root.iterdir():
        if not subdir.is_dir():
            continue
        if names is not None and subdir.name not in names:
            continue
        skill = load_skill(subdir)
        if skill:
            skills.append(skill)
    return skills


def build_skills_prompt(skills: List[Skill]) -> str:
    """Build a prompt fragment listing available skills and their instructions."""
    if not skills:
        return ""
    parts = ["## Available Skills\n"]
    for s in skills:
        parts.append(f"### {s.name}\n{s.content.strip()}\n")
    return "\n".join(parts)
