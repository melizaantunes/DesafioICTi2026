from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces #espaços de ação (aqui serao acoes do tipo discretas)
from ..question_bank import QuestionBank #banco de questoes
from ..student_sim import StudentSim #aluno simulado (parametros+probabilidade de erro+engajamento)

FORMATS = ["short_text", "multiple_choice", "visual", "scaffold"] #formatos de questao que estou utilizando
DIFFICULTIES = [1, 2, 3, 4, 5] #nivel de dificuldade que posso ter

#mapeando a acao a ser tomada pelo agente
def action_to_cell(a: int) -> tuple[str, int]: #transformando a ação no par (formato, dificuldade)
    fmt_i = a // len(DIFFICULTIES) #bloco do formato (0-4, 5-9, 10-14, 15-19)
    d_i = a % len(DIFFICULTIES) #índice da dificuldade dentro do bloco.
    return FORMATS[fmt_i], DIFFICULTIES[d_i]

#classe do ambiente - 20 escolhas (4 formatos × 5 dificuldades)
class FractionTutorEnv(gym.Env):
    
    metadata = {"render_modes": []}

#construtor do ambiente
    def __init__(self, bank_path: str, max_steps: int = 20, seed: int = 0): #20 questoes por sessao
        super().__init__() #inicializando o gym.Env
        self.max_steps = max_steps
        self.bank = QuestionBank(bank_path, seed=seed)
        self.sim = StudentSim(seed=seed) #simulador do aluno

#definindo os limites do vetor de observação
        self.action_space = spaces.Discrete(len(FORMATS) * len(DIFFICULTIES))
        self.observation_space = spaces.Box(
            low=np.array([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([ 3.0, 5.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(seed)

        
        self.t = 0 #contador de passos no episodio
        self.engagement = 1.0 #consideramos que o aluno inicia engajado (engajamento 1)
        self.last_correct = 0.0 #0 para erros, 1 para acertos
        self.last_d = 1 #dificuldade da ultima questao respondida
        self.last_load = 0.2 #carga de leitura do ultimo item

        #guardando o "aluno"
        self.student = None

        #estimativa atual da habilidade do aluno e a incerteza dessa estimativa (diminuindo conforme tutor aprende sobre o aluno)
        self.skill_est = 0.0
        self.skill_unc = 2.0

#reiniciando o episodio
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed) #caso eu queira reproduzir algum episodio

        self.t = 0
        self.engagement = 1.0
        self.last_correct = 0.0
        self.last_d = 1
        self.last_load = 0.2

        self.student = self.sim.sample_student() #criando um novo aluno
        self.skill_est = 0.0 #reiniciando o tutor
        self.skill_unc = 2.0 #começo do 2 pq no começo o tutor nao sabe mto sobre o aluno (incerteza alta)

        return self._obs(), {}

#montando nosso vetor obs
    def _obs(self):
        return np.array([
            self.skill_est,
            self.skill_unc,
            self.engagement,
            self.last_correct,
            (self.last_d - 1) / 4.0,
            self.last_load,
        ], dtype=np.float32)
    
#Atualização da crença do tutor: o ganho base cresce conforme a dificuldade, e acertar o item dificil aumenta mais a habilidade estimada
    def _update_belief(self, d: int, fmt: str, correct: bool):
        gain = 0.18 + 0.05 * (d - 1) #escolhi valores pequenos pra ficar mais estavel e diferenciar dificuldades
        if not correct:
            gain *= -0.12 #a penalização está leve, pra tentar deixar o sistema mais estavel
        # format effect (reading-heavy tasks are noisier)
        load = self.last_load
        gain *= (1.0 - 0.3 * load) #aqui minha ideia foi que questoes com alta leitura sao mais dificeis de avaliar conhecimento, entao dizem menos sobre ter habilidade

        self.skill_est = float(np.clip(self.skill_est + gain, -3.0, 3.0)) #atualizando a habilidade
        self.skill_unc = float(max(0.2, self.skill_unc * 0.96)) #aqui eu deixei um "piso" pra não zerar a habilidade

    def step(self, action: int):
        fmt, d = action_to_cell(int(action))

        # estou forçando o banco de questoes a convergir: se não existe questão naquele formato/dificuldade, eu termino o ep
        if not self.bank.has_cell(fmt, d):
            obs = self._obs()
            return obs, -3.0, True, False, {"reason": "empty_cell", "cell": (fmt, d)}

        item = self.bank.sample(fmt, d) #pegando um item
        load = float(item.reading_load or 0.4) #pegando a carga de leitura do item (se nao tiver nenhuma, eu deixei como 0.4)

#PROBABILIDADE DE ACERTO DO ALUNO NESSE DADO ITEM
        p = self.sim.p_correct(self.student, d, fmt, self.engagement)
        correct = bool(self.rng.random() < p)

        # atualizando engajamento
        self.engagement = self.sim.step_engagement(self.student, d, fmt, self.engagement, correct)

        # atualizando o histórico
        self.last_correct = 1.0 if correct else 0.0
        self.last_d = d
        self.last_load = load
        prev_unc = self.skill_unc #guardando a incerteza
        self._update_belief(d, fmt, correct)

        # recompensa do aprendizado!!! aqui considerei a penalidade menor que o acerto
        r = (1.0 if correct else -0.4)

        # penalizando o abandono da questao (engajamento muito baixo)
        done = False
        truncated = False
        if self.engagement <= 0.12:
            r -= 5.0
            done = True

        r += 0.05 * (prev_unc - self.skill_unc)

        self.t += 1
        if self.t >= self.max_steps:
            truncated = True

        info = {
            "fmt": fmt,
            "difficulty": d,
            "p_correct": float(p),
            "correct": correct,
            "engagement": float(self.engagement),
            "item_id": item.id,
        }
        return self._obs(), float(r), done, truncated, info
