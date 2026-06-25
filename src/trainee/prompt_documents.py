from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


PromptDocumentKind = Literal["project_context", "agent_rules", "constraints", "notes"]


class PromptDocument(BaseModel):
    kind: PromptDocumentKind
    path: str
    text: str
    digest: str
    priority: int


class PromptDocumentLoader:
    _DOCUMENTS = (
        ("context.md", "project_context", 100),
        ("program.md", "agent_rules", 100),
        ("constraints.md", "constraints", 100),
        (".trainee/context.md", "project_context", 200),
        (".trainee/program.md", "agent_rules", 200),
    )

    def load(self, project_root: str | Path) -> list[PromptDocument]:
        root = Path(project_root).expanduser().resolve()
        documents: list[PromptDocument] = []
        for relative_path, kind, priority in self._DOCUMENTS:
            path = root / relative_path
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            documents.append(
                PromptDocument(
                    kind=kind,
                    path=Path(relative_path).as_posix(),
                    text=text,
                    digest=sha256(text.encode("utf-8")).hexdigest(),
                    priority=priority,
                )
            )
        return sorted(documents, key=lambda item: (item.priority, item.path))
