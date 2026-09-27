from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


class TraceLogger:
    def __init__(self, directory: str | Path = "traces") -> None:
        self.directory = Path(directory)
        self._lock = Lock()

    def event(self, run_id: str, event_type: str, **data: Any) -> str:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{run_id}.jsonl"
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "event": event_type,
            **data,
        }
        with self._lock, path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return str(path)
