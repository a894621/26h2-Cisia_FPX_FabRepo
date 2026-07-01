"""
train.py
========

Rôle
----
Script de fine-tuning LoRA d'un petit LLM génératif (SmolLM2-135M par défaut)
sur le dataset « gold » produit par `gold_data.py`. Ce script est la version
exécutable du notebook `demo_finetuning.ipynb` : il peut être lancé sans
Jupyter, dans un pipeline MLflow ou en CI.

Entrées
-------
- gold_data.jsonl  : dataset {instruction, reponse} nettoyé.
- modèle de base   : HuggingFaceTB/SmolLM2-135M (téléchargé localement avant
                     le premier run pour respecter la contrainte on-premise).

Sorties
-------
- ./it_assistant_finetuned/ : adapter LoRA + tokenizer sauvegardés.
- Logs d'inférence avant / après fine-tuning.

Contraintes du brief
--------------------
* Confidentialité : aucun push_to_hub, report_to="none", tout reste local.
* Sobriété       : LoRA (PEFT) sur target_modules attention uniquement.

Usage
-----
    python train.py --data gold_data.jsonl

Brief 6.1 — Cisia Module 3 — Étape 5.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("train")

PROMPT_TEMPLATE = (
    "### Instruction\n"
    "{instruction}\n\n"
    "### Réponse\n"
    "{reponse}"
)

QUESTION_TEST = "Comment réinitialiser mon mot de passe Windows ?"


# ------------------------------------------------------------------
# Data utilities
# ------------------------------------------------------------------
def load_gold(path: Path) -> Dataset:
    """Charge gold_data.jsonl en HF Dataset. Fallback sur quelques exemples
    métier minimaux si le fichier n'existe pas (utile pour un smoke-test)."""
    if path.exists():
        log.info("Chargement du dataset depuis %s", path)
        ds = load_dataset("json", data_files=str(path), split="train")
        return ds

    log.warning("%s introuvable — fallback sur exemples métier en dur", path)
    fallback = [
        {"instruction": "Comment réinitialiser mon mot de passe ?",
         "reponse":     "Rendez-vous sur le portail interne AtosConnect, "
                        "cliquez sur « Mot de passe oublié », validez via "
                        "le SMS reçu sur votre téléphone professionnel."},
        {"instruction": "Mon ordinateur ne démarre plus.",
         "reponse":     "Ouvrez un ticket sur AtosConnect avec la catégorie "
                        "« Poste de travail », précisez le numéro de série "
                        "et joignez une photo de l'écran si possible."},
    ]
    return Dataset.from_list(fallback)


def formatter(example):
    """Transforme une ligne {instruction, reponse} en texte unique pour SFT."""
    return PROMPT_TEMPLATE.format(
        instruction=example["instruction"],
        reponse=example["reponse"],
    )


# ------------------------------------------------------------------
# Inference helper
# ------------------------------------------------------------------
def generate(model, tokenizer, question: str, max_new_tokens: int = 120) -> str:
    prompt = PROMPT_TEMPLATE.format(instruction=question, reponse="").rstrip()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",       type=Path, default=Path("gold_data.jsonl"))
    parser.add_argument("--model",      type=str,  default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--output",     type=Path, default=Path("./it_assistant_finetuned"))
    parser.add_argument("--epochs",     type=int,  default=40)
    parser.add_argument("--batch_size", type=int,  default=1)
    parser.add_argument("--grad_acc",   type=int,  default=4)
    parser.add_argument("--lr",         type=float,default=5e-4)
    parser.add_argument("--max_length", type=int,  default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device détecté : %s", device)

    # ---------- 1. Données ----------
    raw_ds = load_gold(args.data)
    log.info("Dataset chargé : %d exemples", len(raw_ds))

    # Application du template de prompt → colonne "text" attendue par SFTTrainer
    ds = raw_ds.map(
        lambda ex: {"text": formatter(ex)},
        remove_columns=[c for c in raw_ds.column_names if c != "text"],
    )

    # ---------- 2. Modèle + tokenizer ----------
    log.info("Chargement du modèle de base : %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)

    # ---------- 3. Inférence AVANT fine-tuning ----------
    log.info("=== Inférence AVANT fine-tuning ===")
    log.info("%s", generate(model, tokenizer, QUESTION_TEST))

    # ---------- 4. LoRA ----------
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # ---------- 5. SFT ----------
    sft_config = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        max_length=args.max_length,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",                # confidentialité : aucun tracker distant
        push_to_hub=False,                # confidentialité : pas d'upload
        dataset_text_field="text",
        fp16=(device == "cuda"),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    log.info("=== Démarrage de l'entraînement ===")
    trainer.train()

    # ---------- 6. Sauvegarde ----------
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    log.info("Modèle + tokenizer sauvegardés dans %s", args.output)

    # ---------- 7. Inférence APRÈS fine-tuning ----------
    log.info("=== Inférence APRÈS fine-tuning ===")
    log.info("%s", generate(trainer.model, tokenizer, QUESTION_TEST))


if __name__ == "__main__":
    main()
