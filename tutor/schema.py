from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator

Format = Literal["short_text", "multiple_choice", "visual", "scaffold"] #formato das minhas questoes

#representa uma questao do meu banco
class Item(BaseModel):
    # para não quebrar se o JSONL tiver campos a mais (ex.: "answer" vindo do templates)
    model_config = ConfigDict(extra="ignore")

    id: str
    topic: str = "frações"
    format: Format
    difficulty: int = Field(ge=1, le=5)
    variation: int = Field(default=1, ge=1)

    statement: str 
    options: List[str] = Field(default_factory=list)
    correct_index: int = -1 #se nao for multipla escolha fica -1

    solution: str #explicacao curta da questao
    skills: List[str] = Field(default_factory=list) #habilidades envolvidas
    tags: List[str] = Field(default_factory=list)

    # leitura/esforço textual da questao
    reading_load: Optional[float] = None  # quanto maior o valor, maior é o "peso" da leitura

    # validação cruzada: garante coerência entre format/options/correct_index
    @model_validator(mode="after")
    def _validate_mcq_rules(self) -> "Item":
        if self.format == "multiple_choice":
            # MCQ: precisa ter 4 alternativas e correct_index válido (0..3)
            if len(self.options) != 4:
                raise ValueError("Para format='multiple_choice', options deve ter exatamente 4 alternativas.")
            if not (0 <= self.correct_index < len(self.options)):
                raise ValueError("Para format='multiple_choice', correct_index deve estar entre 0 e 3.")
        else:
            # Nao-MCQ: não deve ter alternativas e correct_index deve ser -1
            if len(self.options) != 0:
                raise ValueError("Para formatos não-MCQ, options deve ser lista vazia [].")
            if self.correct_index != -1:
                raise ValueError("Para formatos não-MCQ, correct_index deve ser -1.")
        return self
