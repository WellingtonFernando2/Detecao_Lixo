import argparse
import random
import shutil
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "dataset_pequeno"
YAML_SAIDA = RAIZ / "data" / "dataset_pequeno.yaml"
EXTENSOES = {".jpg", ".jpeg", ".png"}


def carregar_anotadas(origem: Path, nomes_origem, mapa_indices):
    itens = []
    for split in ("train", "valid", "test"):
        imagens = origem / split / "images"
        labels = origem / split / "labels"
        if not imagens.exists():
            continue
        for img in imagens.iterdir():
            if img.suffix.lower() not in EXTENSOES:
                continue
            lbl = labels / (img.stem + ".txt")
            linhas = []
            if lbl.exists():
                for linha in lbl.read_text(encoding="utf-8").splitlines():
                    partes = linha.split()
                    if not partes:
                        continue
                    novo = mapa_indices.get(int(partes[0]))
                    if novo is not None:
                        linhas.append(" ".join([str(novo)] + partes[1:]))
            if linhas:
                itens.append((img, linhas))
    return itens


def gravar(img: Path, linhas, split: str, prefixo: str = ""):
    (SAIDA / "images" / split).mkdir(parents=True, exist_ok=True)
    (SAIDA / "labels" / split).mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, SAIDA / "images" / split / f"{prefixo}{img.name}")
    (SAIDA / "labels" / split / f"{prefixo}{img.stem}.txt").write_text("\n".join(linhas), encoding="utf-8")


def dividir(itens):
    random.seed(0)
    random.shuffle(itens)
    corte = max(1, int(len(itens) * 0.8))
    return itens[:corte], itens[corte:]


def main():
    parser = argparse.ArgumentParser(description="Monta o dataset pequeno para treino")
    parser.add_argument("--roboflow-dir", required=True, help="Pasta do export YOLOv8 do Roboflow (ja extraida)")
    parser.add_argument("--negativos", default=None, help="Pasta com fotos sem nenhum objeto")
    parser.add_argument("--renomear", nargs="*", default=[], help="Corrige nomes: ANTIGO=novo (ex: MTEAL=metal)")
    parser.add_argument("--manter", nargs="*", default=None, help="Classes a manter (nomes ja corrigidos)")
    args = parser.parse_args()

    origem = Path(args.roboflow_dir)
    with open(origem / "data.yaml", encoding="utf-8") as f:
        nomes = yaml.safe_load(f)["names"]
    if isinstance(nomes, dict):
        nomes = [nomes[i] for i in sorted(nomes)]

    renomes = {}
    for par in args.renomear:
        antigo, novo = par.split("=")
        renomes[antigo.lower()] = novo.lower()
    nomes_finais = [renomes.get(str(n).lower(), str(n).lower()) for n in nomes]

    manter = [m.lower() for m in args.manter] if args.manter else list(dict.fromkeys(nomes_finais))
    mapa_indices = {i: manter.index(n) for i, n in enumerate(nomes_finais) if n in manter}

    print(f"Classes do export: {nomes} -> {nomes_finais}")
    print(f"Classes mantidas: {manter}")

    itens = carregar_anotadas(origem, nomes, mapa_indices)
    if not itens:
        print("Nenhuma imagem com as classes escolhidas. Confira --renomear e --manter.")
        return

    if SAIDA.exists():
        shutil.rmtree(SAIDA)

    treino, val = dividir(itens)
    for img, linhas in treino:
        gravar(img, linhas, "train")
    for img, linhas in val:
        gravar(img, linhas, "val")

    contagem = {n: 0 for n in manter}
    for _, linhas in itens:
        for linha in linhas:
            contagem[manter[int(linha.split()[0])]] += 1
    print(f"Imagens anotadas: {len(treino)} treino, {len(val)} validacao")
    print(f"Caixas por classe: {contagem}")

    if args.negativos:
        fotos = [f for f in Path(args.negativos).iterdir() if f.suffix.lower() in EXTENSOES]
        neg_treino, neg_val = dividir([(f, []) for f in fotos])
        for img, _ in neg_treino:
            gravar(img, [], "train", "neg_")
        for img, _ in neg_val:
            gravar(img, [], "val", "neg_")
        print(f"Negativas: {len(neg_treino)} treino, {len(neg_val)} validacao")
        if len(fotos) < 60:
            print("Aviso: poucas fotos negativas. O ideal sao 80 a 120, senao a parede pode voltar a ser detectada.")
    else:
        print("Aviso: sem --negativos, o modelo tende a detectar parede/fundo como objeto.")

    conteudo = {
        "path": str(SAIDA),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(manter)},
    }
    YAML_SAIDA.write_text(yaml.safe_dump(conteudo, allow_unicode=True), encoding="utf-8")
    print(f"Config gerada em {YAML_SAIDA}")
    print("\nProximo passo:")
    print("  python scripts\\train.py --data data\\dataset_pequeno.yaml --epochs 100 --name pequeno")


if __name__ == "__main__":
    main()
