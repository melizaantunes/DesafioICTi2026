from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple #pra deixar o codigo mais claro
from .schema import Item, Format

Cell = Tuple[Format, int]  # a ação a ser tomada tem a ver com a tupla (formato, dificuldade)

#só pra assegurar que o campo seja compativel com o item gerado offline e tambem com o LLM
READING_LOAD = {
    "short_text": 0.20,
    "multiple_choice": 0.35,
    "visual": 0.45,
    "scaffold": 0.70,
}

#organizo os itens do jsonl por celula e amostro de forma aleatoria dentro da celula
#considero um banco estatico, em que o rl escolhe apenas o formato e a dificuldade da questao (e o item especifico é sorteado pra evitar a memorização)
class QuestionBank:
    def __init__(self, path: str | Path, seed: int = 0):
        self.path = Path(path)
        self.rng = random.Random(seed)

        self.items: List[Item] = []
        self.by_cell: Dict[Cell, List[Item]] = {} #pra mapear a chave e o valor/lista de itens naquela celula
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Bank not found: {self.path}") #pra verificar que o arquivo existe

        items: List[Item] = []
        with self.path.open("r", encoding="utf-8") as f: #vou lendo o arquivo linha por linha, remov espacos e quebras de linha
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Fill default reading_load if missing
                obj.setdefault("reading_load", READING_LOAD.get(obj.get("format"), None))
                items.append(Item.model_validate(obj)) #valido e converto pra um item

        self.items = items
        by_cell: Dict[Cell, List[Item]] = {}
        for it in items:
            key: Cell = (it.format, it.difficulty)
            by_cell.setdefault(key, []).append(it)
        self.by_cell = by_cell #ou seja, consigo acessat rapido o item

#uso pra penalizar quando tenho uma celula vazia
    def has_cell(self, fmt: Format, difficulty: int) -> bool:
        return (fmt, difficulty) in self.by_cell and len(self.by_cell[(fmt, difficulty)]) > 0

#sorteio um item
    def sample(self, fmt: Format, difficulty: int) -> Item:
        key: Cell = (fmt, difficulty) #monto a chave
        pool = self.by_cell.get(key)
        if not pool:
            raise KeyError(f"No items for cell={key}. Generate/fill the bank first.")
        return self.rng.choice(pool) #escolho aleaytoriamente um item da lista
