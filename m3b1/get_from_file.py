"""
get_from_file.py
================

Rôle
----
Fusionner deux sources JSONL :
    * `extracted.jsonl`        — issu de la base SQLite (étape 1)
    * `support_rajout.jsonl`   — fichier externe fourni par une autre équipe

Le fichier résultant `merged.jsonl` conserve toutes les paires
(instruction, reponse) et trace l'origine de chacune grâce à un champ
supplémentaire `source` ("sqlite" ou "jsonl_externe").

Entrées
-------
- extracted.jsonl
- support_rajout.jsonl

Sorties
-------
- merged.jsonl : concaténation enrichie d'un champ `source`.

Usage
-----
    python get_from_file.py --extracted extracted.jsonl \
                            --rajout    support_rajout.jsonl \
                            --output    merged.jsonl

Brief 6.1 — Cisia Module 3 — Étape 2.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("get_from_file")


def iter_jsonl(path: Path, source_tag: str) -> Iterable[Dict]:
    """Itère sur un fichier JSONL en taguant la source et en ignorant
    les lignes mal formées (warning loggué)."""
    if not path.exists():
        log.error("Fichier introuvable : %s", path)
        sys.exit(1)

    with path.open("r", encoding="utf-8") as fin:
        for line_num, raw in enumerate(fin, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                log.warning("Ligne %d de %s ignorée (JSON invalide) : %s",
                            line_num, path.name, exc)
                continue

            # On harmonise le schéma minimal attendu
            if "instruction" not in obj or "reponse" not in obj:
                log.warning("Ligne %d de %s ignorée (schéma incomplet)",
                            line_num, path.name)
                continue

            obj["source"] = source_tag
            yield obj


def merge(extracted: Path, rajout: Path, output: Path) -> int:
    """Fusionne les deux JSONL. Retourne le nombre total de lignes écrites."""
    output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output.open("w", encoding="utf-8") as fout:
        for record in iter_jsonl(extracted, source_tag="sqlite"):
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
        for record in iter_jsonl(rajout, source_tag="jsonl_externe"):
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    log.info("Fusion terminée : %d enregistrements écrits dans %s", n, output)
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted", type=Path, default=Path("extracted.jsonl"))
    parser.add_argument("--rajout",    type=Path, default=Path("support_rajout.jsonl"))
    parser.add_argument("--output",    type=Path, default=Path("merged.jsonl"))
    args = parser.parse_args()
    merge(args.extracted, args.rajout, args.output)


if __name__ == "__main__":
    main()
