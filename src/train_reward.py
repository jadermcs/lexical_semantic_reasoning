#!/usr/bin/env python
# coding: utf-8
import argparse
import json
import logging
import os
from string import punctuation
from typing import Iterable

import torch
from datasets import Dataset, DatasetDict
from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder import evaluation, losses, modules
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

import utils
import wandb
from sense_data import mark_target

# Target-word markers. The *usage* (query) is wrapped so the pooling layer can
# restrict to the target span; the *gloss* (document) is left unmarked so the
# same pooling layer falls back to whole-sentence pooling.
M1 = "<t>"
M2 = "</t>"

# Side-marker tokens (added to the vocab like the <t> markers). Each anchor usage
# is prefixed with [QRY] and each gloss with [DOC] so the encoder can tell a
# word-in-context apart from a definition. Note [QRY] sits *outside* the <t> … </t>
# span, so target-word pooling drops it from the sparse query embedding — the query
# asymmetry is the target span itself, so [QRY] is effectively inert; [DOC] is the
# prefix that actually enters the whole-sentence gloss (document) pooling.
QRY = "[QRY]"
DOC = "[DOC]"


class DenseHiddenStatesTransformer(modules.Transformer):
    """Fill-mask transformer that also exposes the dense hidden states.

    The SPLADE head only surfaces the vocab-sized MLM logits as
    ``token_embeddings``. With ``output_hidden_states=True`` in the model
    config, the base class additionally stores every layer under
    ``all_layer_embeddings``; this keeps just the last layer around as
    ``dense_token_embeddings`` so the pooling layer can build the dense
    sentence embedding used by the Barlow Twins loss.
    """

    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        features = super().forward(features, **kwargs)
        all_layers = features.pop("all_layer_embeddings", None)
        if all_layers is not None:
            features["dense_token_embeddings"] = all_layers[-1]
        return features


class TargetWordSpladePooling(modules.SpladePooling):
    """SPLADE pooling restricted to the target-word span.

    Instead of pooling over every token in the sentence, this only pools over
    the tokens that sit strictly between the ``<t>`` / ``</t>`` markers (the
    target word). It does so by swapping the attention mask for a target-word
    mask and then delegating to the regular SPLADE pooling math.
    """

    def __init__(self, *args, m1_tok_id: int, m2_tok_id: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.m1_tok_id = m1_tok_id
        self.m2_tok_id = m2_tok_id

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = features["input_ids"]
        attention_mask = features["attention_mask"]

        is_open = input_ids == self.m1_tok_id
        is_close = input_ids == self.m2_tok_id
        # Positions at/after the opening marker and strictly before the closing
        # marker, excluding the markers themselves.
        after_open = is_open.cumsum(dim=1) > 0
        before_close = is_close.cumsum(dim=1) == 0
        target_mask = after_open & before_close & ~is_open & attention_mask.bool()

        # Fall back to the full sentence for rows where the markers are missing
        # (e.g. truncated out), so we never produce an empty embedding.
        empty = target_mask.sum(dim=1) == 0
        if empty.any():
            target_mask[empty] = attention_mask[empty].bool()

        original_mask = features["attention_mask"]
        features["attention_mask"] = target_mask.to(attention_mask.dtype)

        # Mean-pool the dense (pre-unembedding) hidden states over the target
        # span so we can apply a Barlow Twins loss on the dense representation.
        dense_tokens = features.get("dense_token_embeddings")
        if dense_tokens is not None:
            mask = target_mask.to(dense_tokens.dtype).unsqueeze(-1)
            dense_sum = (dense_tokens * mask).sum(dim=1)
            dense_count = mask.sum(dim=1).clamp(min=1.0)
            features["dense_sentence_embedding"] = dense_sum / dense_count

        features = super().forward(features)
        features["attention_mask"] = original_mask
        return features


class BarlowTwinsLoss:
    """Barlow Twins loss on dense target-word representations.

    Decorrelates feature dimensions across the pair of views (anchor vs
    positive) while pushing the diagonal of the cross-correlation matrix
    to 1. Operates on the pre-unembedding hidden states.
    """

    def __init__(self, lambda_offdiag: float = 5e-3):
        self.lambda_offdiag = lambda_offdiag

    def compute_loss_from_embeddings(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
    ) -> torch.Tensor:
        z_a = z_a.float()
        z_b = z_b.float()
        batch_size, dim = z_a.shape
        z_a_norm = (z_a - z_a.mean(0)) / (z_a.std(0) + 1e-6)
        z_b_norm = (z_b - z_b.mean(0)) / (z_b.std(0) + 1e-6)

        c = (z_a_norm.T @ z_b_norm) / batch_size

        on_diag = (torch.diagonal(c) - 1).pow(2).sum()
        off_diag = c.pow(2).sum() - torch.diagonal(c).pow(2).sum()
        return on_diag + self.lambda_offdiag * off_diag


class StopWordRegularizer:
    def __init__(self, stop_word_ids: list[int]):
        self.stop_word_ids = stop_word_ids

    def compute_loss_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if not self.stop_word_ids:
            return torch.tensor(0.0, device=embeddings.device)
        stop_word_ids = torch.tensor(self.stop_word_ids, device=embeddings.device)
        stop_word_activations = embeddings[:, stop_word_ids]
        return torch.sum(torch.mean(stop_word_activations, dim=0) ** 2)


def loss_forward(
    self,
    sentence_features: Iterable[dict[str, torch.Tensor]],
    labels: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    outputs = [self.model(sentence_feature) for sentence_feature in sentence_features]
    embeddings = [output["sentence_embedding"] for output in outputs]

    losses_dict = {}
    base_loss = self.loss.compute_loss_from_embeddings(embeddings, labels)
    if isinstance(base_loss, dict):
        losses_dict.update(base_loss)
    else:
        losses_dict["base_loss"] = base_loss

    if self.use_document_regularizer_only:
        corpus_loss = self.document_regularizer.compute_loss_from_embeddings(
            torch.cat(embeddings)
        )
    else:
        corpus_loss = self.document_regularizer.compute_loss_from_embeddings(
            torch.cat(embeddings[1:])
        )
    losses_dict["document_regularizer_loss"] = (
        corpus_loss * self.document_regularizer_weight
    )

    if self.query_regularizer_weight is not None:
        query_loss = self.query_regularizer.compute_loss_from_embeddings(embeddings[0])
        losses_dict["query_regularizer_loss"] = (
            query_loss * self.query_regularizer_weight
        )

    if hasattr(self, "stopword_regularizer") and self.stopword_regularizer is not None:
        stopword_loss = self.stopword_regularizer.compute_loss_from_embeddings(
            torch.cat(embeddings)
        )
        losses_dict["stopword_regularizer_loss"] = (
            stopword_loss * self.stopword_regularizer_weight
        )
    if hasattr(self, "barlow_loss") and self.barlow_loss is not None:
        dense_embeddings = [output["dense_sentence_embedding"] for output in outputs]
        if len(dense_embeddings) >= 2:
            barlow = self.barlow_loss.compute_loss_from_embeddings(
                dense_embeddings[0], dense_embeddings[1]
            )
            losses_dict["barlow_loss"] = self.barlow_weight * barlow

    if hasattr(self, "token_sparsity_regularizer_weight"):
        token_loss = sum([output["token_loss"] for output in outputs])
        losses_dict["token_loss"] = self.token_sparsity_regularizer_weight * token_loss

    return losses_dict


def load_sense_fit(prefix: str):
    """Load the fixed lemma-disjoint sense-fit splits written by gen_sense_fit.py.

    Reads ``{prefix}.{train,dev,test}.json``. Each example is
    ``{word, usage, positive, negative}`` (extra metadata is ignored) and becomes
    an ``{anchor, positive, negative}`` triplet. The anchor usage is marked with
    ``<t>`` tags (target-word pooling) and prefixed with ``[QRY]``; the positive
    and negative glosses are prefixed with ``[DOC]`` (full-sentence pooling).
    """
    splits: dict[str, list[dict]] = {}
    for name in ("train", "dev", "test"):
        with open(f"{prefix}.{name}.json") as f:
            examples = json.load(f)
        splits[name] = [
            {
                "anchor": f"{QRY} " + mark_target(ex["usage"], ex["word"]),
                "positive": f"{DOC} " + ex["positive"],
                "negative": f"{DOC} " + ex["negative"],
            }
            for ex in examples
        ]
    return DatasetDict({k: Dataset.from_list(v) for k, v in splits.items()})


def make_evaluator(dataset, name: str):
    # Triplet accuracy: fraction of examples where sim(anchor, positive) beats
    # sim(anchor, negative). Dot similarity matches MNRL's dot_score on the sparse
    # SPLADE embeddings.
    return evaluation.SparseTripletEvaluator(
        anchors=[row["anchor"] for row in dataset],
        positives=[row["positive"] for row in dataset],
        negatives=[row["negative"] for row in dataset],
        name=name,
        similarity_fn_names=["dot"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google-bert/bert-large-uncased")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data", type=str, default="data/sense_fit",
        help="Prefix for the fixed sense-fit splits; reads "
             "{data}.{train,dev,test}.json ({word, usage, positive, negative}).",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--pooling_strategy", type=str, default="sum")
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--loss", type=str, default="mnrl",
                        choices=["mnrl", "triplet"])
    parser.add_argument("--doc_regularization", type=float, default=5e-3)
    parser.add_argument("--query_regularization", type=float, default=5e-3)
    parser.add_argument("--stopword_regularization", type=float, default=5e-4)
    parser.add_argument("--barlow_weight", type=float, default=1e-3)
    parser.add_argument("--barlow_lambda", type=float, default=5e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    utils.set_seed(args.seed)
    logging.info(f"Set seed to: {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    dataset = load_sense_fit(args.data)
    logging.info(
        "Loaded sense-fit triplets: "
        + ", ".join(f"{k}={len(v)}" for k, v in dataset.items())
    )

    num_name = args.run_name or args.model.split("/")[-1]
    wandb.init(
        project="sparse_encoder_custom",
        name=num_name,
        config=vars(args),
    )

    def collate_fn(batch):
        # Tokenize the (anchor, positive, negative) triplet as three columns. The
        # anchor carries [QRY] + <t> markers (target-word pooling); glosses carry
        # [DOC] (whole-sentence pooling). MNRL builds its own in-batch targets, so
        # no labels are needed.
        return {
            "anchor": model.tokenize([row["anchor"] for row in batch]),
            "positive": model.tokenize([row["positive"] for row in batch]),
            "negative": model.tokenize([row["negative"] for row in batch]),
        }

    mlm_transformer = DenseHiddenStatesTransformer(
        args.model,
        max_seq_length=args.max_seq_length,
        transformer_task="fill-mask",
        config_args={
            "hidden_dropout_prob": args.dropout,
            "attention_probs_dropout_prob": args.dropout,
            "output_hidden_states": True,
        },
    )
    # The marker and side tokens must exist before we can build the target-word
    # pooling, so add them to the transformer's tokenizer up front.
    mlm_transformer.tokenizer.add_tokens([M1, M2, QRY, DOC])
    mlm_transformer.auto_model.resize_token_embeddings(len(mlm_transformer.tokenizer))

    mlm_transformer.auto_model.tie_weights()

    m1_tok_id = mlm_transformer.tokenizer.convert_tokens_to_ids(M1)
    m2_tok_id = mlm_transformer.tokenizer.convert_tokens_to_ids(M2)

    splade_pooling = TargetWordSpladePooling(
        pooling_strategy=args.pooling_strategy,
        m1_tok_id=m1_tok_id,
        m2_tok_id=m2_tok_id,
    )
    model = SparseEncoder(
        modules=[mlm_transformer, splade_pooling],
        tokenizer_kwargs={"max_seq_length": args.max_seq_length},
    ).to(device)

    dev_evaluator = make_evaluator(dataset["dev"], name="sensesimx")
    test_evaluator = make_evaluator(dataset["test"], name="sensesimx-test")
    losses.SpladeLoss.forward = loss_forward

    stop_word_ids = []
    if args.stopword_regularization > 0:
        stop_words = ENGLISH_STOP_WORDS | frozenset(punctuation)
        vocab = model.tokenizer.get_vocab()
        stop_word_ids = [
            vocab[stop_word] for stop_word in stop_words if stop_word in vocab
        ]
        logging.info(f"Found {len(stop_word_ids)} stop words in the vocabulary.")

    if args.loss == "triplet":
        # Margin triplet on the single hard negative (no in-batch negatives).
        inner_loss = losses.SparseTripletLoss(model)
    else:  # "mnrl" (default)
        # Contrastive ranking on (anchor, positive, negative): the explicit hard
        # negative plus every other gloss in the batch as in-batch negatives.
        # Default scale=1.0 / dot_score, tuned for large sparse SPLADE scores.
        inner_loss = losses.SparseMultipleNegativesRankingLoss(model)
    # Asymmetric regularization: the document regularizer sparsifies the gloss
    # (document) embeddings, the query regularizer the usage (query) embeddings.
    loss_fn = losses.SpladeLoss(
        model=model,
        loss=inner_loss,
        use_document_regularizer_only=False,
        document_regularizer_weight=args.doc_regularization,
        query_regularizer_weight=args.query_regularization,
    )

    if args.stopword_regularization > 0:
        loss_fn.stopword_regularizer = StopWordRegularizer(stop_word_ids)
        loss_fn.stopword_regularizer_weight = args.stopword_regularization

    if args.barlow_weight > 0:
        loss_fn.barlow_loss = BarlowTwinsLoss(lambda_offdiag=args.barlow_lambda)
        loss_fn.barlow_weight = args.barlow_weight

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    # Calculate total training steps accounting for dynamic batch sizes with hard negatives
    effective_batch_size = args.batch_size
    gradient_accumulation_steps = (
        64 // effective_batch_size if 0 < effective_batch_size <= 64 else 1
    )
    total_steps = args.epochs * (
        len(dataset["train"]) // args.batch_size // gradient_accumulation_steps
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_ratio * total_steps,
        num_training_steps=total_steps,
    )

    output_dir = os.path.join(
        "sparse_encoder",
        os.path.splitext(os.path.basename(args.data))[0],
        args.model.split("/")[-1],
        str(args.seed),
    )
    os.makedirs(output_dir, exist_ok=True)

    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # SPLADE regularizer ramp: quadratic warmup over the first 1/3 of optimizer steps,
    # starting from 0 and reaching the configured weights at the end of warmup.
    # Mirrors SpladeRegularizerWeightSchedulerCallback (default scheduler_type=QUADRATIC,
    # warmup_ratio=1/3) auto-injected by SparseEncoderTrainer.
    max_doc_reg = loss_fn.document_regularizer_weight
    max_query_reg = loss_fn.query_regularizer_weight
    splade_warmup_steps = max(1, int(total_steps * (1 / 3)))
    loss_fn.document_regularizer_weight = 0.0
    if max_query_reg is not None:
        loss_fn.query_regularizer_weight = 0.0

    def update_splade_weights(global_step: int):
        if global_step >= splade_warmup_steps:
            loss_fn.document_regularizer_weight = max_doc_reg
            if max_query_reg is not None:
                loss_fn.query_regularizer_weight = max_query_reg
            return
        ratio = global_step / splade_warmup_steps
        loss_fn.document_regularizer_weight = max_doc_reg * (ratio**2)
        if max_query_reg is not None:
            loss_fn.query_regularizer_weight = max_query_reg * (ratio**2)

    global_step = 0
    best_metric = 0
    run_loss = []
    patience = args.patience
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch + 1}/{args.epochs}")

        # Gradient accumulation target (effective batch of ~64 triplets).
        gradient_accumulation_steps = (
            64 // args.batch_size if 0 < args.batch_size <= 64 else 1
        )

        model.train()
        bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for i, batch in enumerate(bar):
            update_splade_weights(global_step)
            with torch.amp.autocast(model.device.type, enabled=args.fp16):
                features = [
                    {
                        k: v.to(model.device)
                        for k, v in batch[col].items()
                        if k != "modality"
                    }
                    for col in ("anchor", "positive", "negative")
                ]
                # MNRL builds its own in-batch targets, so labels are unused.
                loss = loss_fn(features, None)
            loss = sum(loss.values()) / gradient_accumulation_steps

            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"Warning: Invalid loss detected: {loss.item()}")
                continue

            current_loss = loss.item() * gradient_accumulation_steps
            run_loss.append(current_loss)
            loss.backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                # Calculate gradient norm before clipping
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log training metrics every few steps
                if (i + 1) % (gradient_accumulation_steps * 10) == 0:
                    wandb.log(
                        {
                            "train/loss": current_loss,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "step": (epoch * len(train_dataloader) + i)
                            // gradient_accumulation_steps,
                            "gradient_norm": total_grad_norm,
                        }
                    )

            bar.set_postfix(loss=sum(run_loss[-10:]) / min(10, len(run_loss)))

        # Calculate average loss for this epoch
        epoch_batches = len(list(enumerate(train_dataloader)))
        epoch_avg_loss = (
            sum(run_loss[-epoch_batches:]) / epoch_batches if epoch_batches > 0 else 0
        )
        logging.info(f"Epoch {epoch + 1} finished. Average loss: {epoch_avg_loss:.4f}")
        dev_metrics = dev_evaluator(model)
        logging.info(f"Dev metrics: {dev_metrics}")

        wandb.log({"epoch_avg_loss": epoch_avg_loss, "epoch": epoch + 1, **dev_metrics})

        dev_accuracy = dev_metrics.get(dev_evaluator.primary_metric, 0)
        if dev_accuracy > best_metric:
            best_metric = dev_accuracy
            epochs_without_improvement = 0
            logging.info("New best dev accuracy. Saving model.")
            # model and tokenizer
            model.save_pretrained(os.path.join(output_dir, "best_model"))
        else:
            epochs_without_improvement += 1
            logging.info(
                f"No improvement in dev accuracy for {epochs_without_improvement} epochs."
            )
            if epochs_without_improvement >= patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    logging.info("Training finished. Loading best model for test evaluation.")
    mlm_transformer = DenseHiddenStatesTransformer(
        os.path.join(output_dir, "best_model"),
        transformer_task="fill-mask",
    )
    test_m1_tok_id = mlm_transformer.tokenizer.convert_tokens_to_ids(M1)
    test_m2_tok_id = mlm_transformer.tokenizer.convert_tokens_to_ids(M2)
    splade_pooling = TargetWordSpladePooling(
        pooling_strategy=args.pooling_strategy,
        m1_tok_id=test_m1_tok_id,
        m2_tok_id=test_m2_tok_id,
    )
    model = SparseEncoder(
        modules=[mlm_transformer, splade_pooling],
        device=device,
    )
    model.tokenizer.max_seq_length = args.seq_length
    model.eval()
    model.similarity_fn_name = "dot"

    test_metrics = test_evaluator(model)
    logging.info("Final test metrics:")
    logging.info(test_metrics)

    wandb.log(test_metrics)

    wandb.run.summary.update(
        {
            **test_metrics,
            "total_epochs": args.epochs,
            "model_name": args.model,
            "dataset": args.data,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
        }
    )

    wandb.finish()


if __name__ == "__main__":
    main()
