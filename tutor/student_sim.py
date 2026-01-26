from __future__ import annotations
import math
import random
from dataclasses import dataclass

#a ideia é conectar o formato da questao com a performance do aluno (no sentido de que a prob de acerto pra alunos com problema de leitura é menor e o engajamento é mais 
#rapido em itens mais "pesados" em relacao a texto)
FORMAT_LOAD = {
    "short_text": 0.20,
    "multiple_choice": 0.35,
    "visual": 0.45,
    "scaffold": 0.70,
}

#definindo o quao favoravel é o item para o aluno
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

#perfil do aluno simulado:
@dataclass
class StudentParams:
    theta: float              # usei o theta pra representar a habilidade latente do assunto (no caso aqui, frações)
    reading_sensitivity: float # o quanto o texto atrapalha a performance e o engajamento do aluno
    noise: float = 0.15       # quanto aleatoriedade tem nas respostas

class StudentSim:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def sample_student(self) -> StudentParams: #crio um novo aluno e sorteio cada parametro definido anteriormente
        theta = self.rng.uniform(-1.5, 1.5)
        rs = self.rng.uniform(0.0, 2.0)
        return StudentParams(theta=theta, reading_sensitivity=rs)

#convertendo a dificuldade em um "numero" (pra depois subtrair no acerto da questao)
    def item_difficulty_bias(self, d: int) -> float:
        return (d - 3) * 0.6
    
#probabilidade de acerto no item
    def p_correct(self, params: StudentParams, d: int, fmt: str, engagement: float) -> float:
        load = FORMAT_LOAD.get(fmt, 0.4)
        #montando o score do aluno
        x = (params.theta
             - self.item_difficulty_bias(d)
             - load * params.reading_sensitivity
             + 0.8 * (engagement - 0.5))
        # add noise by jittering x
        x += self.rng.gauss(0.0, params.noise)
        return sigmoid(x) ##pra transformar o score em probabilidade

#atualizando o engagamento apos resolver a questao
    def step_engagement(self, params: StudentParams, d: int, fmt: str, engagement: float, correct: bool) -> float:
        load = FORMAT_LOAD.get(fmt, 0.4)
        # o delta representa uma queda do engajamento devido a leitura e a dificuldade da questao (e tambem acrescenta o efeito do erro e acerto)
        delta = -0.10 * load - 0.03 * (d - 1)
        if correct:
            delta += 0.06
        else:
            delta -= 0.02
        # aqui criei uma penalidade extra caso o aluno tenha alta sensibilidade e o item tenha muito texto
        delta -= 0.04 * load * params.reading_sensitivity
        new_e = max(0.0, min(1.0, engagement + delta))
        return new_e
