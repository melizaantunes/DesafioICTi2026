from __future__ import annotations  
import argparse 
import json      #para ler o arquivo de log (JSON) e transformar em objetos Python
from pathlib import Path  
from collections import Counter  #contador para ocorrências por categoria


##a ideia dessr aquivo era gerar um relatorio ao fim da sessao e devolver ao aluno na forma de um textinho pronto
def template_report(session_log: list[dict]) -> str:
    #conta quantas vezes cada formato aparece no log
    fmt_counts = Counter(ev["fmt"] for ev in session_log if "fmt" in ev)

    #conta acertos por formato e total por formato
    corr_by_fmt = Counter()
    tot_by_fmt = Counter()

    #percorre cada questão respondida
    for ev in session_log:
        fmt = ev.get("fmt")  #pega o formato do evento
        if not fmt:
            continue  #se não tiver formato ignora esse evento

        tot_by_fmt[fmt] += 1 

        if ev.get("correct"):  
            corr_by_fmt[fmt] += 1 

    #MONTANDO O TEXTO
    lines = []  # lista de linhas do relatório
    lines.append("Relatório da sessão (protótipo Desafio bolsista ICTi 2026)")
    lines.append("") 
    lines.append(f"Total de itens: {len(session_log)}")  #quantidade total de eventos
    lines.append("")

    #se houve pelo menos um formato contado
    if fmt_counts:
        lines.append("Distribuição por formato:")

        
        #ordenando do mais frequente p/ o menos frequente
        for fmt, n in fmt_counts.most_common():
            #acuracia=acertos/total
            acc = (corr_by_fmt[fmt] / tot_by_fmt[fmt]) if tot_by_fmt[fmt] else 0.0
            #formatando para linha no formato: "- scaffold: 5 itens | acerto ~ 60%"
            lines.append(f"- {fmt}: {n} itens | acerto ~ {acc:.0%}")

        lines.append("")


    #assumindo que questoes do tipo "visual" e "scaffold" tendem a ser mais pesados
    heavy = sum(fmt_counts[f] for f in ["visual", "scaffold"])

    #se teve muitos itens pesados (>=3 ou >= 1/3 do total), escreve uma obs
    if heavy >= max(3, len(session_log)//3):
        lines.append("Observação: muitos itens com maior carga de leitura (visual/scaffold).")
        lines.append("Se houver queda de engajamento, tente alternar com itens short_text ou múltipla escolha.")
        lines.append("")

    #SUGESTOES GERAIS (SEMPRE DEVOLVO ESSA INFO (poderia posteriormente pensar em adaptar esse texto conforme o resultado do aluno))
    lines.append("Próximos passos sugeridos:")
    lines.append("- Revisar habilidades com mais erros e repetir com dificuldade 1-2.")
    lines.append("- Aumentar dificuldade gradualmente quando o acerto ficar acima de ~70%.")
    return "\n".join(lines)

def main():

    ap = argparse.ArgumentParser() 
    #caminho do arquivo JSON que contém a lista de step infos
    ap.add_argument("--log", type=str, required=True, help="Path to a JSON list of step infos.")
    ap.add_argument("--out", type=str, default="runs/report.txt")
    args = ap.parse_args()  #le os argumentos do terminal
    #convertendo de JSON para objeto python
    log = json.loads(Path(args.log).read_text(encoding="utf-8"))
    #gero o texto
    text = template_report(log)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    #escrevendo o relatório em .txt
    out.write_text(text, encoding="utf-8")
    print(f"Wrote report -> {out}")

if __name__ == "__main__":
    main()  # executa o main apenas se esse arquivo for rodado diretamente (não importado)
