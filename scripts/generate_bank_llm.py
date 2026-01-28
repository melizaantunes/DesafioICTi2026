from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field


#Modelo do LLM
class LLMExercise(BaseModel):
    statement: str #me devolve o enunciado
    options: List[str] = Field(default_factory=list) #caso seja de multipla escolha, as alternativas
    correct_index: int = -1 #indice da alternativa correta (ou -1 quando nao for de alternativa)
    solution: str #explicação curta
    skills: List[str] = Field(default_factory=list) #habilidades envolvidas 
    tags: List[str] = Field(default_factory=list) #tags


#configurando meu banco de questoes
TOPIC = "frações"
FORMATS = {
    "short_text": "Enunciado curto + conta direta (sem alternativas).",
    "multiple_choice": "Múltipla escolha com 4 alternativas.",
    "visual": "Questão com representação visual descrita em texto (sem imagem real).",
    "scaffold": "Resolução guiada em passos (scaffold), com lacunas/etapas."
}
DIFFICULTIES = [1, 2, 3, 4, 5]
N_VARIATIONS = 2  #escolhi pequeno


#montando o prompt
def build_prompt(fmt: str, difficulty: int, variation: int) -> str:
    fmt_desc = FORMATS[fmt]

    # Regras específicas por formato
    if fmt == "multiple_choice":
        format_rules = (
            "Gere 4 alternativas (options) e marque correct_index (0..3). "
            "As alternativas devem ser plausíveis (inclua erros comuns)."
        )
    else:
        format_rules = (
            "Não gere alternativas (options deve ser lista vazia) "
            "e correct_index deve ser -1."
        )

    return f"""
Você é um gerador de questões educativas.
Crie UMA questão de matemática sobre {TOPIC} em português (Brasil).

Formato: {fmt} -> {fmt_desc}
Dificuldade: {difficulty} (1 = muito fácil, 5 = mais difícil)
Variação: {variation} (apenas para diversificar exemplos)

Regras:
- A questão deve ser autocontida (sem depender de outras).
- Evite números enormes.
- Forneça uma solução curta e correta.
- Inclua skills e tags úteis.

Regras de saída:
- Responda APENAS com JSON válido seguindo o schema.
- {format_rules}

Observações para dificuldade:
- D=1: frações simples, mesma base/denominador pequeno.
- D=3: equivalência/simplificação + comparação.
- D=5: operações mistas, problemas contextualizados curtos, atenção a armadilhas comuns.
""".strip()


#usando gemini para o json estruturado
def generate_one(client: genai.Client, model_name: str, fmt: str, difficulty: int, variation: int) -> LLMExercise:
    prompt = build_prompt(fmt, difficulty, variation)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": LLMExercise.model_json_schema(),
        },
    )

    # usando o pydantic para evitar respostas vazias
    if not getattr(response, "text", None):
        raise RuntimeError("Resposta vazia do modelo (sem JSON).")

    return LLMExercise.model_validate_json(response.text)


#salvando JSONL
def main():
    load_dotenv()

    # lendo a key do api
    client = genai.Client()
    model_name = "gemini-3-flash-preview"

    out_path = Path("data/items_seed.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # arquivo separado para erros se precisar avaliar depois
    err_path = Path("data/items_seed_errors.jsonl")
    err_path.parent.mkdir(parents=True, exist_ok=True)

    items_written = 0
    errors_written = 0

    #abrindo dois arquivos: um para itens válidos e outro para logs de erro
    with out_path.open("w", encoding="utf-8") as f, err_path.open("w", encoding="utf-8") as ferr:
        for fmt in FORMATS.keys():
            for d in DIFFICULTIES:
                for v in range(1, N_VARIATIONS + 1):
                    try:
                        ex = generate_one(client, model_name, fmt, d, v)

                        item = {
                            "id": f"{TOPIC}_{fmt}_d{d}_v{v}",
                            "topic": TOPIC,
                            "format": fmt,
                            "difficulty": d,
                            "variation": v,
                            "statement": ex.statement,
                            "options": ex.options,
                            "correct_index": ex.correct_index,
                            "solution": ex.solution,
                            "skills": ex.skills,
                            "tags": ex.tags,
                        }

                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        items_written += 1
                        print(f"[OK] {item['id']}")

                        #pequena pausa para evitar rate limit
                        time.sleep(0.3)

                    except Exception as e:
                        msg = str(e)
                        print(f"[ERRO] fmt={fmt}, d={d}, v={v} -> {e}")

                        #salvando erro em JSONL separado (para rastreabilidade)
                        error_record = {
                            "status": "error",
                            "topic": TOPIC,
                            "format": fmt,
                            "difficulty": d,
                            "variation": v,
                            "model": model_name,
                            "error": msg,
                            "ts_unix": time.time(),
                        }
                        ferr.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                        errors_written += 1

                        #problema de quota
                        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                            print("Quota atingida. Encerrando geração para não desperdiçar tentativas.")
                            break  #sai do loop mais interno

                #garante que não continue iterando sem chance de sucesso
                if "msg" in locals() and ("RESOURCE_EXHAUSTED" in str(msg) or "429" in str(msg)):
                    break
            if "msg" in locals() and ("RESOURCE_EXHAUSTED" in str(msg) or "429" in str(msg)):
                break

    print(f"\nConcluído. Itens gerados: {items_written}")
    print(f"Erros registrados: {errors_written}")
    print(f"Arquivo (itens): {out_path.resolve()}")
    print(f"Arquivo (erros): {err_path.resolve()}")


if __name__ == "__main__":
    main()
