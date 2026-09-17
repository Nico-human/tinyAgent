from typing import TypedDict

import yaml

from .custom_context import WorkspaceContext


class Skill(TypedDict):
    name: str
    description: str
    content: str


def parse_frontmatter(text: str) -> tuple[dict, str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        return {}, text

    closing_index = next(
        (
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.rstrip("\r\n") == "---"
        ),
        None,
    )
    if closing_index is None:
        return {}, text

    frontmatter = "".join(lines[1:closing_index])
    body = "".join(lines[closing_index + 1 :]).strip()
    try:
        metadata = yaml.safe_load(frontmatter) or {}
    except yaml.YAMLError:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    return metadata, body


def load_workspace_skills(workspace: WorkspaceContext) -> dict[str, Skill]:
    """Read a workspace's skills once when constructing the application context."""
    skill_path = workspace.root / "skills"
    skills: dict[str, Skill] = {}
    if not skill_path.exists():
        return skills
    skills_root = skill_path.resolve()
    for manifest in sorted(skill_path.glob("*/SKILL.md")):
        if not manifest.is_file() or not manifest.resolve().is_relative_to(skills_root):
            continue
        content = manifest.read_text(encoding=workspace.encoding)
        metadata, body = parse_frontmatter(content)
        raw_name = metadata.get("name")
        name = raw_name.strip() if isinstance(raw_name, str) else ""
        name = name or manifest.parent.name
        raw_description = metadata.get("description")
        description = (
            raw_description.strip() if isinstance(raw_description, str) else ""
        )
        description = description or body.split("\n", 1)[0]
        description = " ".join(str(description).lstrip("# ").split())
        skills[name] = {
            "name": name,
            "description": description,
            "content": content,
        }
    return skills
