from dataclasses import dataclass


@dataclass
class SemanticBlock:
    text: str
    block_type: str
    start_offset: int
    end_offset: int
