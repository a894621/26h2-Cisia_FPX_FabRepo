"""
gold_data.py
============

Rôle
----
Produire le jeu de données « gold » prêt pour le fine-tuning à partir
du fichier fusionné `merged.jsonl`.

Étapes de nettoyage appliquées
------------------------------
1. Strip des espaces et normalisation des sauts de ligne (\r\n -> \n).
2. Suppression des entrées dont `instruction` ou `reponse` est vide / None.
3. Validation des longueurs minimales :
        - instruction >= 5 caractères
        - reponse     >= 10 caractères
4. Déduplication exacte basée sur (instruction.lower().strip(),
                                   reponse.lower().strip()).
5. Statistiques imprimées en fin de traitement.

Entrées
-------
- merged.jsonl

Sorties
-------
- gold_data.jsonl : dataset propre, prêt pour SFTTrainer.

Usage
-----
    python gold_data.py --input merged.jsonl --output gold_data.jsonl

Brief 6.1 — Cisia Module 3 — Étape 3.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("gold_data")

MIN_INSTRUCTION_LEN = 5
MIN_REPONSE_LEN     = 10


def normalize_text(text: str) -> str:
    """Strip + normalisation des sauts de ligne."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def clean(input_path: Path, output_path: Path) -> dict:
    if not input_path.exists():
        log.error("Fichier d'entrée introuvable : %s", input_path)
        sys.exit(1)

    stats = {
        "total":      0,
        "empties":    0,
        "too_short":  0,
        "duplicates": 0,
        "kept":       0,
    }
    seen = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line_num, raw in enumerate(fin, start=1):
            raw = raw.strip()
            if not raw:
                continue
            stats["total"] += 1

            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                log.warning("Ligne %d : JSON invalide ignoré (%s)", line_num, exc)
                continue

            instruction = normalize_text(obj.get("instruction") or "")
            reponse     = normalize_text(obj.get("reponse")     or "")

            # 2. valeurs manquantes
            if not instruction or not reponse:
                stats["empties"] += 1
                continue

            # 3. longueur minimale
            if len(instruction) < MIN_INSTRUCTION_LEN \
               or len(reponse) < MIN_REPONSE_LEN:
                stats["too_short"] += 1
                continue

            # 4. doublons
            key = (instruction.lower(), reponse.lower())
            if key in seen:
                stats["duplicates"] += 1
                continue
            seen.add(key)

            clean_record = {
                "instruction": instruction,
                "reponse":     reponse,
            }
            # On conserve la source si présente (utile pour audit)
            if "source" in obj:
                clean_record["source"] = obj["source"]

            fout.write(json.dumps(clean_record, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    log.info("=== Statistiques de nettoyage ===")
    log.info("Entrées lues          : %d", stats["total"])
    log.info("Entrées vides         : %d", stats["empties"])
    log.info("Entrées trop courtes  : %d", stats["too_short"])
    log.info("Doublons supprimés    : %d", stats["duplicates"])
    log.info("Entrées conservées    : %d", stats["kept"])
    log.info("Fichier produit       : %s", output_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",  type=Path, default=Path("merged.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("gold_data.jsonl"))
    args = parser.parse_args()
    clean(args.input, args.output)


if __name__ == "__main__":
    main()
