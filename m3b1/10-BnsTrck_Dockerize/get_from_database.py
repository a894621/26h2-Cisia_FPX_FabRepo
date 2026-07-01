"""
get_from_database.py
====================

Rôle
----
Extraire les paires (instruction, reponse) de la base SQLite `support_it.db`
(table `tickets`) et les sérialiser dans un fichier JSONL prêt à être
fusionné avec d'autres sources de données.

Entrées
-------
- support_it.db  : base SQLite contenant la table `tickets`
                   (colonnes : id, instruction, reponse).

Sorties
-------
- extracted.jsonl : un objet JSON par ligne, encodé en UTF-8,
                    de la forme :
                        {"instruction": "...", "reponse": "..."}

Usage
-----
    python get_from_database.py --db support_it.db --output extracted.jsonl

Brief 6.1 — Cisia Module 3 — Étape 1.
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("get_from_database")


def extract(db_path: Path, output_path: Path) -> int:
    """Extrait les paires (instruction, reponse) et les écrit en JSONL.

    Retourne le nombre de lignes valides écrites.
    """
    if not db_path.exists():
        log.error("Base SQLite introuvable : %s", db_path)
        sys.exit(1)

    log.info("Connexion à la base SQLite : %s", db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # On filtre directement en SQL les lignes inutilisables (NULL ou vides)
    query = """
        SELECT id, instruction, reponse
        FROM tickets
        WHERE instruction IS NOT NULL
          AND reponse     IS NOT NULL
          AND TRIM(instruction) <> ''
          AND TRIM(reponse)     <> ''
    """
    cursor.execute(query)

    n_written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for row in cursor.fetchall():
            record = {
                "instruction": row["instruction"].strip(),
                "reponse":     row["reponse"].strip(),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    conn.close()
    log.info("Extraction terminée : %d enregistrements écrits dans %s",
             n_written, output_path)
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db",     type=Path, default=Path("support_it.db"),
                        help="Chemin vers la base SQLite (défaut : support_it.db)")
    parser.add_argument("--output", type=Path, default=Path("extracted.jsonl"),
                        help="Fichier JSONL de sortie (défaut : extracted.jsonl)")
    args = parser.parse_args()
    extract(args.db, args.output)


if __name__ == "__main__":
    main()
