# import json
# from pathlib import Path
# from datasets import Dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
# from peft import LoraConfig
# from trl import DPOTrainer
# from workspace.utils.logging_utils import get_logger

# logger = get_logger("rlhf.train_dpo_lora")

# def load_dpo_dataset(path: str, max_rows: int | None = None) -> Dataset:
#     rows = []
#     with open(path, "r", encoding="utf-8") as f:
#         for i, line in enumerate(f):
#             rows.append(json.loads(line))
#             if max_rows and len(rows) >= max_rows:
#                 break
#     return Dataset.from_list(rows)

# def train_dpo_lora(
#     dpo_jsonl: str,
#     base_model_name: str,
#     output_dir: str,
#     max_rows: int = 500,
#     batch_size: int = 2,
#     grad_accum: int = 8,
#     lr: float = 1e-5,
#     epochs: int = 1,
#     beta: float = 0.1,
# ) -> None:
#     outp = Path(output_dir)
#     outp.mkdir(parents=True, exist_ok=True)

#     ds = load_dpo_dataset(dpo_jsonl, max_rows=max_rows)
#     logger.info(f"Loaded DPO rows: {len(ds)}")

#     tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")
#     ref_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")

#     peft_config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM",
#         target_modules=["q_proj","k_proj","v_proj","o_proj"]  # works for many decoder models
#     )

#     args = TrainingArguments(
#         output_dir=str(outp),
#         per_device_train_batch_size=batch_size,
#         gradient_accumulation_steps=grad_accum,
#         learning_rate=lr,
#         num_train_epochs=epochs,
#         logging_steps=10,
#         save_strategy="epoch",
#         fp16=False,
#         bf16=True,
#         report_to=[],
#     )

#     def tokenize_row(row):
#         # DPOTrainer expects raw strings for prompt/chosen/rejected; it will tokenize.
#         return row

#     trainer = DPOTrainer(
#         model=model,
#         ref_model=ref_model,
#         args=args,
#         #beta=beta,
#         train_dataset=ds.map(tokenize_row),
#         #tokenizer=tokenizer,
#         peft_config=peft_config,
#         #max_prompt_length=1024,
#         #max_length=1152,
#     )

#     trainer.train()
#     trainer.save_model(str(outp))
#     logger.info(f"Saved DPO LoRA -> {outp}")

import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType

from trl import DPOTrainer, DPOConfig


def _load_jsonl(path: str, max_rows: int | None = None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(json.loads(line))
    return rows


def train_dpo_lora(
    dpo_jsonl: str,
    base_model_name: str,
    output_dir: str,
    max_rows: int = 2000,
    batch_size: int = 2,
    grad_accum: int = 8,
    lr: float = 1e-5,
    epochs: int = 1,
    beta: float = 0.1,
    max_prompt_length: int = 1024,
    max_length: int = 1152,
):
    """
    Expects JSONL rows with keys: prompt, chosen, rejected
    Saves LoRA adapter (and trainer artifacts) to output_dir
    """
    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(dpo_jsonl, max_rows=max_rows)

    # Minimal schema enforcement
    for r in rows[:5]:
        if not all(k in r for k in ("prompt", "chosen", "rejected")):
            raise ValueError("DPO dataset must contain keys: prompt, chosen, rejected")

    ds = Dataset.from_list(rows)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # LoRA config (small + safe)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # common for LLaMA/Mistral-family
    )

    dpo_args = DPOConfig(
        output_dir=str(outp),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=10,
        save_steps=200,
        bf16=torch.cuda.is_available(),
        fp16=False,
        beta=beta,
        max_prompt_length=max_prompt_length,
        max_length=max_length,
        remove_unused_columns=False,
        report_to=[],
    )

    # Newer TRL uses processing_class instead of tokenizer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(str(outp))

    # Save a copy of config for reproducibility
    with open(outp / "dpo_config_used.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model_name": base_model_name,
                "beta": beta,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "lr": lr,
                "epochs": epochs,
                "max_prompt_length": max_prompt_length,
                "max_length": max_length,
            },
            f,
            indent=2,
        )

    return str(outp)

