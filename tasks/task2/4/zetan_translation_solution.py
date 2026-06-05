import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

try:
    import sacrebleu
except ImportError as exc:
    raise ImportError(
        "Missing dependency `sacrebleu`. Install it with: pip install sacrebleu"
    ) from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path, has_dst: bool) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if has_dst:
                rows.append({"src": obj["src"], "dst": obj["dst"]})
            else:
                rows.append({"src": obj["src"]})
    return rows


def maybe_slice(rows: List[Dict[str, str]], max_samples: Optional[int]) -> List[Dict[str, str]]:
    if max_samples is None or max_samples <= 0:
        return rows
    return rows[:max_samples]


class PairDataset(Dataset):
    def __init__(self, rows: List[Dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.rows[idx]


@dataclass
class Collator:
    tokenizer: AutoTokenizer
    max_source_length: int
    max_target_length: int
    prefix: str

    def train_collate(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        src_texts = [self.prefix + x["src"] for x in batch]
        tgt_texts = [x["dst"] for x in batch]

        model_inputs = self.tokenizer(
            src_texts,
            truncation=True,
            max_length=self.max_source_length,
            padding=True,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            text_target=tgt_texts,
            truncation=True,
            max_length=self.max_target_length,
            padding=True,
            return_tensors="pt",
        )["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        model_inputs["labels"] = labels
        return model_inputs

    def infer_collate(self, batch: List[Dict[str, str]]) -> Dict[str, List[str]]:
        src_texts = [self.prefix + x["src"] for x in batch]
        return {"src_texts": src_texts}


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def generate_texts(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    device: torch.device,
    max_source_length: int,
    max_new_tokens: int,
    num_beams: int,
    length_penalty: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> List[str]:
    model.eval()
    predictions: List[str] = []

    for batch in tqdm(dataloader, desc="Generating", leave=False):
        encoded = tokenizer(
            batch["src_texts"],
            truncation=True,
            max_length=max_source_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = move_to_device(encoded, device)
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        predictions.extend([x.strip() for x in texts])

    return predictions


@torch.no_grad()
def evaluate_bleu(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    val_rows: List[Dict[str, str]],
    collator: Collator,
    device: torch.device,
    batch_size: int,
    max_source_length: int,
    max_new_tokens: int,
    num_beams: int,
    length_penalty: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> float:
    val_loader = DataLoader(
        PairDataset(val_rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator.infer_collate,
    )
    preds = generate_texts(
        model=model,
        tokenizer=tokenizer,
        dataloader=val_loader,
        device=device,
        max_source_length=max_source_length,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    refs = [x["dst"] for x in val_rows]
    return float(sacrebleu.corpus_bleu(preds, [refs]).score)


def save_submission(
    test_rows: List[Dict[str, str]],
    test_preds: List[str],
    submission_path: Path,
) -> None:
    with submission_path.open("w", encoding="utf-8") as f:
        for row, pred in zip(test_rows, test_preds):
            record = {"src": row["src"], "dst": pred}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    if args.fp16 and args.bf16:
        raise ValueError("Use only one mixed precision mode: --fp16 or --bf16.")

    data_dir = Path(args.data_dir)
    test_path = data_dir / args.test_file

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.inference_only:
        checkpoint_dir = (
            Path(args.checkpoint_dir)
            if args.checkpoint_dir is not None
            else Path(args.output_dir) / "best_model"
        )
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {checkpoint_dir}. "
                "Pass --checkpoint-dir or run training first."
            )
        test_rows = load_jsonl(test_path, has_dst=False)
        print(f"Inference-only mode. Test size: {len(test_rows)}")
        print(f"Loading checkpoint from: {checkpoint_dir}")

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir).to(device)

        collator = Collator(
            tokenizer=tokenizer,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            prefix=args.source_prefix,
        )
        test_loader = DataLoader(
            PairDataset(test_rows),
            batch_size=args.predict_batch_size,
            shuffle=False,
            collate_fn=collator.infer_collate,
        )
        test_preds = generate_texts(
            model=model,
            tokenizer=tokenizer,
            dataloader=test_loader,
            device=device,
            max_source_length=args.max_source_length,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        submission_path = Path(args.submission_path)
        save_submission(test_rows, test_preds, submission_path)
        print(f"Submission saved: {submission_path} ({len(test_preds)} lines)")
        return

    train_path = data_dir / args.train_file
    val_path = data_dir / args.val_file

    train_rows = maybe_slice(load_jsonl(train_path, has_dst=True), args.max_train_samples)
    val_rows = maybe_slice(load_jsonl(val_path, has_dst=True), args.max_val_samples)
    test_rows = load_jsonl(test_path, has_dst=False)
    print(f"Train size: {len(train_rows)}, Val size: {len(val_rows)}, Test size: {len(test_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(device)

    collator = Collator(
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        prefix=args.source_prefix,
    )

    train_loader = DataLoader(
        PairDataset(train_rows),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collator.train_collate,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = max(1, steps_per_epoch * args.num_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = torch.cuda.is_available()
    if args.bf16 and use_cuda and hasattr(torch.cuda, "is_bf16_supported"):
        if not torch.cuda.is_bf16_supported():
            print("CUDA bf16 is not supported on this GPU. Falling back to fp16/fp32.")
            args.bf16 = False
    use_bf16 = bool(args.bf16 and use_cuda)
    use_fp16 = bool(args.fp16 and use_cuda and not use_bf16)
    autocast_enabled = use_bf16 or use_fp16
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler(device=autocast_device, enabled=use_fp16)

    best_bleu = -1.0
    epochs_without_improvement = 0
    best_dir = Path(args.output_dir) / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")
        finite_steps = 0
        skipped_non_finite = 0
        for step, batch in enumerate(pbar, start=1):
            batch = move_to_device(batch, device)

            with torch.amp.autocast(
                device_type=autocast_device,
                enabled=autocast_enabled,
                dtype=autocast_dtype,
            ):
                loss = model(**batch).loss / args.gradient_accumulation_steps

            if not torch.isfinite(loss):
                skipped_non_finite += 1
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix(
                    loss="nan",
                    step=global_step,
                    skipped=skipped_non_finite,
                )
                continue

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            finite_steps += 1

            should_step = (
                step % args.gradient_accumulation_steps == 0 or step == len(train_loader)
            )
            if should_step:
                prev_scale = scaler.get_scale() if use_fp16 else None
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if use_fp16:
                    step_was_applied = scaler.get_scale() >= prev_scale
                else:
                    step_was_applied = True
                if step_was_applied:
                    scheduler.step()
                    global_step += 1

            running_loss = epoch_loss / max(1, finite_steps)
            pbar.set_postfix(
                loss=f"{running_loss:.4f}",
                step=global_step,
                skipped=skipped_non_finite,
            )

        train_loss = epoch_loss / max(1, finite_steps)
        model.config.use_cache = True
        val_bleu = evaluate_bleu(
            model=model,
            tokenizer=tokenizer,
            val_rows=val_rows,
            collator=collator,
            device=device,
            batch_size=args.eval_batch_size,
            max_source_length=args.max_source_length,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_BLEU={val_bleu:.2f}")
        if args.gradient_checkpointing:
            model.config.use_cache = False

        if val_bleu > best_bleu + args.min_delta:
            best_bleu = val_bleu
            epochs_without_improvement = 0
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"Saved new best model to {best_dir} (BLEU={best_bleu:.2f})")
        else:
            epochs_without_improvement += 1
            print(
                f"No BLEU improvement for {epochs_without_improvement} epoch(s). "
                f"Best BLEU is {best_bleu:.2f}."
            )
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered (patience={args.patience}).")
                break

    print(f"Best validation BLEU: {best_bleu:.2f}")

    print("Loading best checkpoint for test inference...")
    tokenizer = AutoTokenizer.from_pretrained(best_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(best_dir).to(device)

    test_loader = DataLoader(
        PairDataset(test_rows),
        batch_size=args.predict_batch_size,
        shuffle=False,
        collate_fn=collator.infer_collate,
    )
    test_preds = generate_texts(
        model=model,
        tokenizer=tokenizer,
        dataloader=test_loader,
        device=device,
        max_source_length=args.max_source_length,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    submission_path = Path(args.submission_path)
    save_submission(test_rows, test_preds, submission_path)
    print(f"Submission saved: {submission_path} ({len(test_preds)} lines)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Zetan->English translation model and build JSONL submission."
    )
    parser.add_argument("--data-dir", type=str, default="ml_trainings.alien_translation")
    parser.add_argument("--train-file", type=str, default="train")
    parser.add_argument("--val-file", type=str, default="val")
    parser.add_argument("--test-file", type=str, default="test_no_reference")

    parser.add_argument("--model-name", type=str, default="google/byt5-small")
    parser.add_argument("--source-prefix", type=str, default="translate Zetan to English: ")

    parser.add_argument("--output-dir", type=str, default="outputs_zetan_byt5")
    parser.add_argument("--submission-path", type=str, default="submission_zetan_translation.jsonl")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default=None)

    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--predict-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.05)

    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
