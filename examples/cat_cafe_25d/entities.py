from dataclasses import dataclass

@dataclass
class Cat:
    ident: str
    foot_y: float
    body_y: float
    container: int = 0
    x: float = 0
    state: str = 'idle'
