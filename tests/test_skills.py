from pathlib import Path

from langclaw.tools.skills import build_skills_prompt, load_skill, load_skills


def test_load_skill_reads_description_from_heading(tmp_path: Path):
    skill_dir = tmp_path / "summarize"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "# Summarize docs\n\nUse concise bullets.", encoding="utf-8"
    )

    skill = load_skill(skill_dir)
    assert skill is not None
    assert skill.name == "summarize"
    assert skill.description == "Summarize docs"
    assert "Use concise bullets." in skill.content


def test_load_skills_filters_by_requested_names(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "SKILL.md").write_text("# A", encoding="utf-8")
    (b / "SKILL.md").write_text("# B", encoding="utf-8")

    skills = load_skills(tmp_path, names=["b"])
    assert [s.name for s in skills] == ["b"]


def test_build_skills_prompt_empty_and_populated(tmp_path: Path):
    assert build_skills_prompt([]) == ""
    skill_dir = tmp_path / "research"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Research\nFind sources", encoding="utf-8")
    skill = load_skill(skill_dir)
    assert skill is not None

    prompt = build_skills_prompt([skill])
    assert "## Available Skills" in prompt
    assert "### research" in prompt
    assert "Find sources" in prompt
