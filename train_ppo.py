##treino do ppo
from __future__ import annotations
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from tutor.envs.fraction_tutor_env import FractionTutorEnv #importando meu tutor

#pra poder usar no dummyvecenv
def make_env(bank: str, seed: int):
    def _thunk():
        return FractionTutorEnv(bank_path=bank, max_steps=20, seed=seed)
    return _thunk

#definindo o treino
def main():
    ap = argparse.ArgumentParser() 
    ap.add_argument("--bank", type=str, default="data/items_bank.jsonl")
    ap.add_argument("--timesteps", type=int, default=200_000) #aqui escolhi 200 passos
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="models/ppo_20actions.zip") #pra salvar os treinos
    args = ap.parse_args()

#criando o env vetorizado e anormalizacao
    env = DummyVecEnv([make_env(args.bank, args.seed)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.0)

#criando o modelo ppo
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=1,
        n_steps=1024, #rollout
        batch_size=256,
        gamma=0.99, #fator de desconto
        learning_rate=3e-4, #taxa de aprendizado
    )
    model.learn(total_timesteps=args.timesteps) #treino

    Path("models").mkdir(exist_ok=True)

    #garantindo que a pasta do --out existe (caso o usuario mude o caminho)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    model.save(args.out)

    #salvando as estatisticas com nome ligado ao modelo (evita vecnormalize errado)
    vn_path = Path(args.out).with_suffix(".vecnormalize.pkl")
    env.save(str(vn_path)) #salvando as estatisticas

    print(f"Saved model -> {args.out}")
    print(f"Saved VecNormalize -> {vn_path}")

if __name__ == "__main__":
    main()
