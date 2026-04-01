from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Set

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from tokenizers.decoders import BPEDecoder


DEFAULT_SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]


LANGUAGE_KEYWORDS: Set[str] = {
	# Python
	"False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
	"def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in",
	"is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
	# Java/C/C++/C#/JS/TS common
	"abstract", "boolean", "byte", "case", "catch", "char", "const", "default", "do", "double", "enum",
	"extends", "final", "float", "goto", "implements", "instanceof", "int", "interface", "long", "native",
	"new", "package", "private", "protected", "public", "short", "static", "strictfp", "super", "switch",
	"synchronized", "this", "throw", "throws", "transient", "void", "volatile", "null", "let", "var",
	"function", "typeof", "delete", "undefined", "constructor", "namespace", "module", "require", "export",
	"readonly", "declare", "type", "keyof", "infer", "never", "unknown", "any",
	# Go
	"chan", "defer", "fallthrough", "func", "go", "map", "range", "select", "struct", "type",
	# Rust
	"crate", "dyn", "extern", "impl", "match", "mod", "move", "mut", "pub", "ref", "self", "Self",
	"trait", "unsafe", "use", "where", "loop", "union",
	# SQL
	"SELECT", "FROM", "WHERE", "GROUP", "BY", "ORDER", "INSERT", "UPDATE", "DELETE", "JOIN", "LEFT",
	"RIGHT", "INNER", "OUTER", "ON", "AS", "DISTINCT", "LIMIT", "OFFSET", "CREATE", "DROP", "ALTER",
	# Misc literals
	"true", "false",
}


TOKEN_PATTERN = re.compile(
	r"""
	[A-Za-z_][A-Za-z0-9_]*
	|\d+\.\d+
	|\d+
	|==|!=|<=|>=|<<|>>|\+\+|--|->|=>|&&|\|\|
	|[{}\[\]().,;:+\-*/%&|^~!<>?=:@#\\]
	""",
	re.VERBOSE,
)


def code_tokenize_line(line: str) -> List[str]:
	"""
	Regex-based code tokenizer.
	- Keeps language keywords as whole tokens.
	- Splits operators/punctuation into separate tokens.
	"""
	if not isinstance(line, str):
		line = "" if line is None else str(line)

	tokens = TOKEN_PATTERN.findall(line)
	return tokens


def _boolize(value) -> bool:
	if isinstance(value, bool):
		return value
	if isinstance(value, (int, float)):
		return bool(value)
	if value is None:
		return False
	return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _iter_csv_rows(csv_path: str):
	with open(csv_path, "r", encoding="utf-8", newline="") as fin:
		reader = csv.DictReader(fin)
		for row in reader:
			yield row


def iter_code_corpus_from_csv(
	csv_path: str,
	split: str = "train",
	drop_comment: bool = False,
	drop_blank: bool = True,
) -> Iterable[str]:
	"""
	Yield whitespace-joined token sequences from LineDP CSV.
	"""
	if split not in {"all", "train", "test"}:
		raise ValueError("split must be one of {'all', 'train', 'test'}")

	for row in _iter_csv_rows(csv_path):
		if "code_line" not in row:
			raise ValueError(f"CSV must contain 'code_line' column: {csv_path}")

		if "is_test_file" in row and _boolize(row.get("is_test_file")):
			continue

		if drop_comment and "is_comment" in row and _boolize(row.get("is_comment")):
			continue
		if drop_blank and "is_blank" in row and _boolize(row.get("is_blank")):
			continue

		line = "" if row.get("code_line") is None else str(row.get("code_line"))
		tokens = code_tokenize_line(line)
		if not tokens:
			continue
		yield " ".join(tokens)


def iter_code_corpus_from_csv_dir(
	csv_dir: str,
	split: str = "train",
	drop_comment: bool = False,
	drop_blank: bool = True,
) -> Iterable[str]:
	"""
	Yield tokenized lines from all CSV files under a directory.
	"""
	directory = Path(csv_dir)
	if not directory.exists() or not directory.is_dir():
		raise FileNotFoundError(f"csv_dir does not exist or is not a directory: {csv_dir}")

	csv_files = sorted(directory.glob("*.csv"))
	if not csv_files:
		raise FileNotFoundError(f"No CSV files found under: {csv_dir}")

	for csv_path in csv_files:
		for item in iter_code_corpus_from_csv(
			csv_path=str(csv_path),
			split=split,
			drop_comment=drop_comment,
			drop_blank=drop_blank,
		):
			yield item


def train_bpe_vocab_from_csv(
	csv_path: str,
	output_dir: str,
	vocab_size: int = 20000,
	min_frequency: int = 2,
	split: str = "train",
	drop_comment: bool = False,
	drop_blank: bool = True,
	extra_keywords: Sequence[str] | None = None,
) -> Tokenizer:
	"""
	Train BPE tokenizer from LineDP code_line column.

	To avoid subword split for language keywords, keywords are inserted into
	trainer.special_tokens, so they are kept as atomic tokens.
	"""
	output = Path(output_dir)
	output.mkdir(parents=True, exist_ok=True)

	keywords = set(LANGUAGE_KEYWORDS)
	if extra_keywords:
		keywords.update(extra_keywords)

	tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
	tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
	tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
	tokenizer.decoder = BPEDecoder()

	special_tokens = list(DEFAULT_SPECIAL_TOKENS) + sorted(keywords)
	trainer = trainers.BpeTrainer(
		vocab_size=vocab_size,
		min_frequency=min_frequency,
		special_tokens=special_tokens,
	)

	corpus_iter = iter_code_corpus_from_csv(
		csv_path=csv_path,
		split=split,
		drop_comment=drop_comment,
		drop_blank=drop_blank,
	)
	tokenizer.train_from_iterator(corpus_iter, trainer=trainer)

	tokenizer_json = output / "tokenizer.json"
	tokenizer.save(str(tokenizer_json))
	tokenizer.model.save(str(output))

	with open(output / "keywords.txt", "w", encoding="utf-8") as fout:
		for kw in sorted(keywords):
			fout.write(kw + "\n")

	return tokenizer


def train_bpe_vocab_from_csv_dir(
	csv_dir: str,
	output_dir: str,
	vocab_size: int = 30000,
	min_frequency: int = 2,
	split: str = "train",
	drop_comment: bool = False,
	drop_blank: bool = True,
	extra_keywords: Sequence[str] | None = None,
) -> Tokenizer:
	"""
	Train one unified BPE tokenizer from all CSV files in a directory.
	"""
	output = Path(output_dir)
	output.mkdir(parents=True, exist_ok=True)

	keywords = set(LANGUAGE_KEYWORDS)
	if extra_keywords:
		keywords.update(extra_keywords)

	tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
	tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
	tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
	tokenizer.decoder = BPEDecoder()

	special_tokens = list(DEFAULT_SPECIAL_TOKENS) + sorted(keywords)
	trainer = trainers.BpeTrainer(
		vocab_size=vocab_size,
		min_frequency=min_frequency,
		special_tokens=special_tokens,
	)

	corpus_iter = iter_code_corpus_from_csv_dir(
		csv_dir=csv_dir,
		split=split,
		drop_comment=drop_comment,
		drop_blank=drop_blank,
	)
	tokenizer.train_from_iterator(corpus_iter, trainer=trainer)

	tokenizer_json = output / "tokenizer.json"
	tokenizer.save(str(tokenizer_json))
	tokenizer.model.save(str(output))

	with open(output / "keywords.txt", "w", encoding="utf-8") as fout:
		for kw in sorted(keywords):
			fout.write(kw + "\n")

	with open(output / "source_csv_dir.txt", "w", encoding="utf-8") as fout:
		fout.write(str(Path(csv_dir).resolve()) + "\n")

	return tokenizer


def load_bpe_tokenizer(model_dir: str) -> Tokenizer:
	tokenizer_json = Path(model_dir) / "tokenizer.json"
	if not tokenizer_json.exists():
		raise FileNotFoundError(f"tokenizer.json not found under {model_dir}")
	return Tokenizer.from_file(str(tokenizer_json))


def _build_cli() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="CodeDiff tokenizer utilities")
	subparsers = parser.add_subparsers(dest="command", required=True)

	train_parser = subparsers.add_parser("train-bpe", help="Train BPE from LineDP CSV")
	train_parser.add_argument("--csv", required=True, help="Path to LineDP CSV")
	train_parser.add_argument("--out", required=True, help="Output directory for tokenizer files")
	train_parser.add_argument("--vocab-size", type=int, default=20000)
	train_parser.add_argument("--min-frequency", type=int, default=2)
	train_parser.add_argument("--split", choices=["all", "train", "test"], default="train")
	train_parser.add_argument("--drop-comment", action="store_true")
	train_parser.add_argument("--keep-blank", action="store_true")

	train_dir_parser = subparsers.add_parser("train-bpe-dir", help="Train unified BPE from all CSV files in a directory")
	train_dir_parser.add_argument("--csv-dir", required=True, help="Directory containing LineDP CSV files")
	train_dir_parser.add_argument("--out", required=True, help="Output directory for tokenizer files")
	train_dir_parser.add_argument("--vocab-size", type=int, default=30000)
	train_dir_parser.add_argument("--min-frequency", type=int, default=2)
	train_dir_parser.add_argument("--split", choices=["all", "train", "test"], default="train")
	train_dir_parser.add_argument("--drop-comment", action="store_true")
	train_dir_parser.add_argument("--keep-blank", action="store_true")

	inspect_parser = subparsers.add_parser("inspect-line", help="Show tokenization for one code line")
	inspect_parser.add_argument("--text", required=True)

	return parser


def main() -> None:
	parser = _build_cli()
	args = parser.parse_args()

	if args.command == "train-bpe":
		tokenizer = train_bpe_vocab_from_csv(
			csv_path=args.csv,
			output_dir=args.out,
			vocab_size=args.vocab_size,
			min_frequency=args.min_frequency,
			split=args.split,
			drop_comment=args.drop_comment,
			drop_blank=not args.keep_blank,
		)
		print(f"trained tokenizer vocab_size={tokenizer.get_vocab_size()} -> {args.out}")
		return

	if args.command == "train-bpe-dir":
		tokenizer = train_bpe_vocab_from_csv_dir(
			csv_dir=args.csv_dir,
			output_dir=args.out,
			vocab_size=args.vocab_size,
			min_frequency=args.min_frequency,
			split=args.split,
			drop_comment=args.drop_comment,
			drop_blank=not args.keep_blank,
		)
		print(f"trained unified tokenizer vocab_size={tokenizer.get_vocab_size()} -> {args.out}")
		return

	if args.command == "inspect-line":
		print(code_tokenize_line(args.text))
		return

	raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
	main()

