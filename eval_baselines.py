from __future__ import annotations
import argparse #rodar pelo terminal
from dataclasses import dataclass
import json
from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from tutor.envs.fraction_tutor_env import FractionTutorEnv

##resultado de um episodio
@dataclass
class EpisodeResult:
    return_sum: float #soma total das recompensas do episodio
    abandoned: bool #se o aluno abandonou (baixo engaj)
    steps: int #numero de questoes feitas
    completed: bool
    reason: str | None

#rodando um ep com uma politica
def run_episode(env: FractionTutorEnv, policy_fn, rng: random.Random, seed: int) -> EpisodeResult:
    obs, _ = env.reset(seed=seed) #pegando obs inicial
    ret = 0.0 #retorno acumulado
    abandoned = False #caso tenha abandono
    steps = 0 #passos
    done = False #terminou real
    truncated = False #terminou por limite de passos
    last_info = {}

    while not (done or truncated):
        a = policy_fn(obs, rng) #politica escolhe uma acao e recebe uma nova obs e a recompensa
        obs, r, done, truncated, info = env.step(a)
        last_info = info if isinstance(info, dict) else {}
        ret += float(r)
        steps += 1

    reason = last_info.get("reason", None)
    if reason is None:
        if truncated:
            reason = "time_limit"
        elif done:
            reason = "low_engagement"

    abandoned = (reason == "low_engagement")
    completed = (reason == "time_limit")

    return EpisodeResult(
        return_sum=ret,
        abandoned=abandoned,
        steps=steps,
        completed=completed,
        reason=reason,
    ) #resumo do episodio


##BASELINES

#aleatorio
def policy_random(obs, rng: random.Random) -> int:
    return rng.randrange(20) #pois tenho 20 tipos de acoes diferentes

#escadinha
def policy_staircase(obs, rng: random.Random) -> int:
    # aumenta dificuldade se acertou a ultima, diminui se errou a ultima
    last_correct = float(obs[3])
    last_d_norm = float(obs[4])
    d = int(round(last_d_norm * 4)) + 1
    if last_correct >= 0.5:
        d = min(5, d + 1)
    else:
        d = max(1, d - 1)
    fmt = "multiple_choice"  # fixei o formato como multipla escolha
    fmt_i = ["short_text", "multiple_choice", "visual", "scaffold"].index(fmt)
    return fmt_i * 5 + (d - 1)

#consciente de engajamento
def policy_engagement(obs, rng: random.Random) -> int:
    # se engajamento estiver baixo ou carga estiver alta, reduz dificuldade para evitar abandono
    engagement = float(obs[2])
    last_load = float(obs[5])
    last_correct = float(obs[3])
    last_d_norm = float(obs[4])

    #quando o engajamento está caindo, prioriza itens mais leves e fáceis
    if engagement < 0.35 or last_load > 0.70:
        d = 1 if engagement < 0.25 else 2
        fmt = "multiple_choice"
    else:
        # caso contrário, faz a escadinha padrão
        d = int(round(last_d_norm * 4)) + 1
        if last_correct >= 0.5:
            d = min(5, d + 1)
        else:
            d = max(1, d - 1)
        fmt = "multiple_choice"
    fmt_i = ["short_text", "multiple_choice", "visual", "scaffold"].index(fmt)
    return fmt_i * 5 + (d - 1)

#carregando o PPO e a normalizacao
def load_ppo(model_path: str, bank: str) -> tuple[PPO, VecNormalize]:
    def _thunk():
        return FractionTutorEnv(bank_path=bank, max_steps=20, seed=0)
    venv = DummyVecEnv([_thunk])
    venv = VecNormalize.load("models/vecnormalize.pkl", venv)
    #desligando modo treino (pra nao atualizar estatisticas e nao normalizar a recompensa)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(model_path, env=venv)
    return model, venv

#execucao e metricas
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=str, default="data/items_bank.jsonl")
    ap.add_argument("--model", type=str, default="models/ppo_20actions.zip")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="runs/eval")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    res = {}

    # avaliando as baselines
    for name, pol in [("random", policy_random), ("staircase", policy_staircase), ("engagement", policy_engagement)]:
        eps = []
        for i in range(args.episodes):
            env = FractionTutorEnv(bank_path=args.bank, max_steps=20, seed=args.seed + i)
            rng = random.Random(args.seed + 10_000 + i)
            eps.append(run_episode(env, pol, rng, seed=args.seed + i))
        res[name] = eps #guardando os episodios pra cada baseline

    # avaliando o PPO
    try:
        model, venv = load_ppo(args.model, args.bank)

        ##definindo a politica que usa o PPO pra escolher a ação
        def policy_ppo(obs, rng):
            obs_arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
            obs_norm = venv.normalize_obs(obs_arr)
            a, _ = model.predict(obs_norm, deterministic=True)
            return int(np.asarray(a).item())

        eps = []
        for i in range(args.episodes):
            env2 = FractionTutorEnv(bank_path=args.bank, max_steps=20, seed=args.seed + 1000 + i)
            rng = random.Random(args.seed + 20_000 + i)
            eps.append(run_episode(env2, policy_ppo, rng, seed=args.seed + 1000 + i))
        res["ppo"] = eps
    except Exception as e:
        print(f"[WARN] Could not load PPO model: {e}")

    # salvando as metricas
    out_json = Path(args.outdir) / "summary.json"
    #resumo
    def summarize(eps):
        arr = np.array([e.return_sum for e in eps], dtype=float)
        ab = np.array([1.0 if e.abandoned else 0.0 for e in eps], dtype=float)
        st = np.array([e.steps for e in eps], dtype=float)
        comp = np.array([1.0 if e.completed else 0.0 for e in eps], dtype=float)

        reasons = {}
        for e in eps:
            key = e.reason if e.reason is not None else "unknown"
            reasons[key] = reasons.get(key, 0) + 1

        return {
            "mean_return": float(arr.mean()), #retorno medio
            "std_return": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, #variabilidade
            "abandon_rate": float(ab.mean()), #taxa de abandono
            "mean_steps": float(st.mean()), #media de passos
            "complete_rate": float(comp.mean()),
            "reasons": reasons,
        }

    summary = {k: summarize(v) for k, v in res.items()}
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_json}")

    # histograma
    plt.figure()
    for k, eps in res.items():
        plt.hist([e.return_sum for e in eps], bins=20, alpha=0.5, label=k)
    plt.xlabel("Episode return")
    plt.ylabel("Count")
    plt.legend()
    out_png = Path(args.outdir) / "returns_hist.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_png}")

if __name__ == "__main__":
    main()

