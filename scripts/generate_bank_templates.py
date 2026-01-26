from __future__ import annotations
import argparse
import json
import math #pq preciso fazer contas de mmc em questoes scaffold
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Tuple

FORMATS = ["short_text", "multiple_choice", "visual", "scaffold"] #formatos possiveis de questao
DIFFICULTIES = [1, 2, 3, 4, 5] #dificuldades

READING_LOAD = {
    "short_text": 0.20,
    "multiple_choice": 0.35,
    "visual": 0.45,
    "scaffold": 0.70,
}
##aqui é pra transformar a fração do exercicio numa string (ex: fraction(1,2) vira 1/2)
def frac_str(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}"

##escolho dois denominadores (para a e b) a partir de um pool que depende da dificuldade, sendo que denominadores maiores são mais dificeis de lidar (fui até 20)
def choose_denoms(rng: random.Random, difficulty: int) -> Tuple[int, int]:
    if difficulty <= 2:
        pool = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif difficulty == 3:
        pool = [4, 5, 6, 7, 8, 9, 10, 12, 14, 15]
    else:
        pool = [6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20]
    return rng.choice(pool), rng.choice(pool)

#empacotando os dados
@dataclass
class OpExample:
    a: Fraction
    b: Fraction
    op: str
    result: Fraction

#escolhendo se vamos somar, subtrair, multiplicar e dividir e realizando a operacao
def build_op_example(rng: random.Random, d: int) -> OpExample:
    op = rng.choice(["add", "sub"] if d <= 2 else ["add", "sub", "mul", "div"])
    da, db = choose_denoms(rng, d)
    a = Fraction(rng.randint(1, da - 1), da)
    b = Fraction(rng.randint(1, db - 1), db)

    if op == "add":
        res = a + b
    elif op == "sub":
        res = a - b
    elif op == "mul":
        res = a * b
    else:
        res = a / b
    return OpExample(a=a, b=b, op=op, result=res)

#gerando o exemplo base (montando enunciado)
def make_short_text(rng: random.Random, d: int) -> dict:
    ex = build_op_example(rng, d)

    sym = {"add": "+", "sub": "−", "mul": "×", "div": "÷"}[ex.op]
    stmt = f"Calcule: {frac_str(ex.a)} {sym} {frac_str(ex.b)}."
    skills = {
        "add": ["soma de frações"],
        "sub": ["subtração de frações"],
        "mul": ["multiplicação de frações"],
        "div": ["divisão de frações"],
    }[ex.op]

    ans = frac_str(ex.result.limit_denominator())
    sol = f"Resposta: {ans}. (Simplifique a fração se necessário.)"
    return {
        "statement": stmt,
        "answer": ans,
        "solution": sol,
        "skills": skills,
        "tags": ["frações", "template"],
        "options": [],
        "correct_index": -1,
        "_op": ex.op,
        "_a": frac_str(ex.a),
        "_b": frac_str(ex.b),
    }

##jamais podemos dividir por zero (dica da prof):)
def _safe_fraction(n: int, d: int) -> Fraction | None:
    if d == 0:
        return None
    return Fraction(n, d)

##definindo as questoes de multipla escolha
def make_mcq(rng: random.Random, d: int) -> dict:
    base = make_short_text(rng, d)
    a = Fraction(*map(int, base["_a"].split("/")))
    b = Fraction(*map(int, base["_b"].split("/")))
    op = base["_op"]
    correct = Fraction(*map(int, base["answer"].split("/")))



    distractors = set() #respostas erradas nas demais alternativas

##aqui eu resolvi definir uma função que adiciona uma alternativa se (nao for igual a correta, o denominador nao for zero e nao for negativa (aqui apenas para facilitar))
    def add_candidate(fr: Fraction | None):
        if fr is None:
            return
        if fr == correct:
            return
        if fr.denominator <= 0:
            return
        if fr < 0:
            return
        distractors.add(frac_str(fr.limit_denominator()))

    # erros comuns que vemos em sala de aula
    if op in ("add", "sub"):
        # denominadores diferentes e o aluno soma ou subtrai diretamente numerador e denominador
        wrong1 = _safe_fraction(a.numerator + (b.numerator if op == "add" else -b.numerator),
                                a.denominator + b.denominator)
        add_candidate(wrong1)
        # denominadores diferentes e o aluno soma ou subtrai apenas os denominadores
        wrong2 = _safe_fraction(a.numerator + (b.numerator if op == "add" else -b.numerator),
                                a.denominator)
        add_candidate(wrong2)
    elif op == "mul":
        # multiplica numeradores mas mantém denominadores
        wrong1 = _safe_fraction(a.numerator * b.numerator, a.denominator)
        add_candidate(wrong1)
    else:  # div
        # multiplica ao invez de dividir
        wrong1 = a * b
        add_candidate(wrong1)

    # fazendo perturbações pra criar distratores
    for delta_n, delta_d in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1)]:
        cand = _safe_fraction(correct.numerator + delta_n, correct.denominator + delta_d)
        add_candidate(cand)

    # se por algum motivo ainda não tiver 3 alternativas, preenchemos com frações aleatórias plausíveis
    tries = 0
    while len(distractors) < 3 and tries < 200:
        tries += 1
        dn = max(1, correct.denominator + rng.choice([-2, -1, 1, 2, 3]))
        nn = max(0, correct.numerator + rng.choice([-3, -2, -1, 1, 2, 3]))
        cand = _safe_fraction(nn, dn)
        add_candidate(cand)

    # usando tambem algumas respostas "padrao" como distratores
    fallback = ["0/1", "1/2", "2/1", "3/2", "2/3", "3/4"]
    for fb in fallback:
        if len(distractors) >= 3:
            break
        if fb != base["answer"]:
            distractors.add(fb)

    opts = list(distractors)[:3] + [base["answer"]]
    rng.shuffle(opts)

    base["options"] = opts
    base["correct_index"] = opts.index(base["answer"])
    base["statement"] = base["statement"].replace("Calcule:", "Escolha a alternativa correta para:") + " (resposta em forma de fração)"
    base["skills"] = base["skills"] + ["múltipla escolha"]

    # remove internal fields
    base.pop("_op", None)
    base.pop("_a", None)
    base.pop("_b", None)
    return base

##fazendo as questoes do tipo "visual" - aqui pensei em 2 tipos de questões: figura com partes pintadas ou comparação de duas frações (equivalencia)
def make_visual(rng: random.Random, d: int) -> dict:
    if d <= 2:
        n = rng.choice([4, 6, 8, 10])
        shaded = rng.randint(1, n - 1)
        stmt = (
            f"Imagine uma barra dividida em {n} partes iguais. "
            f"{shaded} partes estão pintadas. Qual fração da barra está pintada?"
        )
        ans = frac_str(Fraction(shaded, n))
        sol = f"A fração é {shaded}/{n}, que pode ser simplificada para {ans}."
        skills = ["representação visual", "fração parte-todo", "simplificação"]
    else:
        n1 = rng.choice([6, 8, 10, 12])
        n2 = rng.choice([6, 8, 10, 12])
        s1 = rng.randint(1, n1 - 1)
        s2 = rng.randint(1, n2 - 1)
        f1 = Fraction(s1, n1)
        f2 = Fraction(s2, n2)
        rel = "maior" if f1 > f2 else ("menor" if f1 < f2 else "igual")
        stmt = (
            f"Considere duas barras: a primeira tem {s1} de {n1} partes pintadas, "
            f"e a segunda tem {s2} de {n2} partes pintadas. "
            "A primeira fração pintada é maior, menor ou igual à segunda?"
        )
        ans = rel
        sol = f"Comparando {frac_str(f1)} e {frac_str(f2)}, a primeira é {rel} que a segunda."
        skills = ["comparação de frações", "representação visual"]
    return {
        "statement": stmt,
        "answer": ans,
        "solution": sol,
        "skills": skills,
        "tags": ["frações", "visual", "template"],
        "options": [],
        "correct_index": -1,
    }

#definindo questoes do tipo scaffold (completar espaços em brancos dando um passo a passo)
def make_scaffold(rng: random.Random, d: int) -> dict:
    da, db = choose_denoms(rng, max(2, d))
    a = Fraction(rng.randint(1, da - 1), da)
    b = Fraction(rng.randint(1, db - 1), db)
    res = a + b

    lcm = abs(da * db) // math.gcd(da, db)
    a2 = Fraction(a.numerator * (lcm // da), lcm)
    b2 = Fraction(b.numerator * (lcm // db), lcm)

    stmt = (
        "Resolva passo a passo (preencha mentalmente as lacunas):\n"
        f"1) Encontre o MMC de {da} e {db}: MMC = ___.\n"
        f"2) Reescreva {frac_str(a)} e {frac_str(b)} com denominador {lcm}.\n"
        f"   {frac_str(a)} = ___/{lcm} e {frac_str(b)} = ___/{lcm}.\n"
        f"3) Some os numeradores: ___ + ___ = ___.\n"
        f"4) Resultado: ___/{lcm}. Simplifique se possível."
    )

    ans = frac_str(res)
    sol = (
        f"MMC({da},{db}) = {lcm}. "
        f"{frac_str(a)} = {frac_str(a2)}, {frac_str(b)} = {frac_str(b2)}. "
        f"Soma: {a2.numerator}+{b2.numerator}={a2.numerator + b2.numerator}. "
        f"Resultado: {frac_str(a2 + b2)} (simplificado: {ans})."
    )
    return {
        "statement": stmt,
        "answer": ans,
        "solution": sol,
        "skills": ["MMC", "equivalência de frações", "soma de frações", "scaffold"],
        "tags": ["frações", "scaffold", "template"],
        "options": [],
        "correct_index": -1,
    }

##aqui chamo qual função chamar (das def acima) dependendo do tormato
def generate_one(rng: random.Random, fmt: str, d: int) -> dict:
    if fmt == "short_text":
        return make_short_text(rng, d)
    if fmt == "multiple_choice":
        return make_mcq(rng, d)
    if fmt == "visual":
        return make_visual(rng, d)
    return make_scaffold(rng, d)

#(definindo a CLI + geração do JSONL)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/items_bank.jsonl")
    ap.add_argument("--n_per_cell", type=int, default=10, help="items per (format,difficulty)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.append and out.exists() else "w"
    with out.open(mode, encoding="utf-8") as f:
        for fmt in FORMATS:
            for d in DIFFICULTIES:
                for v in range(1, args.n_per_cell + 1):
                    payload = generate_one(rng, fmt, d)
                    item = {
                        "id": f"frações_{fmt}_d{d}_t{args.seed}_v{v}",
                        "topic": "frações",
                        "format": fmt,
                        "difficulty": d,
                        "variation": v,
                        "statement": payload["statement"],
                        "options": payload.get("options", []),
                        "correct_index": payload.get("correct_index", -1),
                        "answer": payload.get("answer"),
                        "solution": payload["solution"],
                        "skills": payload.get("skills", []),
                        "tags": payload.get("tags", []),
                        "reading_load": READING_LOAD.get(fmt, None),
                    }
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote -> {out.resolve()}")

if __name__ == "__main__":
    main()
