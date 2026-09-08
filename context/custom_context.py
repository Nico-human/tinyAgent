import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspaceContext:
    root: Path
    encoding: str = "utf-8" # 先写死

    def __post_init__(self):
        object.__setattr__(self, "root", self.root.resolve())

@dataclass(frozen=True)
class BlockCommandContext:
    deny_list: tuple[str, ...] = ("rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=")
    destructive_command_word: re.Pattern[str] = re.compile(
        r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
    )
    destructive: tuple[str, ...] = ("rm ", "> /etc/", "chmod 777")
