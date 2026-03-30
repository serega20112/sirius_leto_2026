from dataclasses import dataclass
from datetime import datetime
from typing import List

@dataclass
class Student:
    id: str
    name: str
    group_name: str
    photo_paths: List[str]
    created_at: datetime