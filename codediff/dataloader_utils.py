from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class CodeLineExample:
    filename: str
    line_number: int
    code_line: str
    label: int
    is_test_file: bool
    is_comment: bool
    is_blank: bool
    file_label: bool


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def read_line_dp_csv(
    csv_path: str,
    split: str = "all",
    drop_comment: bool = False,
    drop_blank: bool = False,
) -> List[CodeLineExample]:
    required = {
        "filename",
        "is_test_file",
        "code_line",
        "line_number",
        "is_comment",
        "is_blank",
        "file-label",
        "line-label",
    }

    if split not in {"all", "train", "test"}:
        raise ValueError("split must be one of {'all', 'train', 'test'}")

    examples: List[CodeLineExample] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        header = set(reader.fieldnames or [])
        missing = required.difference(header)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            is_test = _to_bool(row.get("is_test_file"))
            if is_test:
                continue

            is_comment = _to_bool(row.get("is_comment"))
            if drop_comment and is_comment:
                continue

            is_blank = _to_bool(row.get("is_blank"))
            if drop_blank and is_blank:
                continue

            line_number_raw = row.get("line_number")
            try:
                line_number = int(line_number_raw) if line_number_raw not in (None, "") else 0
            except ValueError:
                line_number = 0

            code_line = "" if row.get("code_line") is None else str(row.get("code_line"))

            examples.append(
                CodeLineExample(
                    filename=str(row.get("filename", "")),
                    line_number=line_number,
                    code_line=code_line,
                    label=1 if _to_bool(row.get("line-label")) else 0,
                    is_test_file=is_test,
                    is_comment=is_comment,
                    is_blank=is_blank,
                    file_label=_to_bool(row.get("file-label")),
                )
            )

    return examples


class CodeLineDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_seq_len: int = 256,
        split: str = "all",
        drop_comment: bool = False,
        drop_blank: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = read_line_dp_csv(
            csv_path=csv_path,
            split=split,
            drop_comment=drop_comment,
            drop_blank=drop_blank,
        )
        self.pad_id = self._resolve_pad_id()

    def _resolve_pad_id(self) -> int:
        for token in ("<pad>", "[PAD]"):
            idx = self.tokenizer.token_to_id(token)
            if idx is not None:
                return idx
        return 0

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        encoded = self.tokenizer.encode(ex.code_line)
        token_ids = encoded.ids[: self.max_seq_len]
        attention_mask = [1] * len(token_ids)
        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "label": ex.label,
            "filename": ex.filename,
            "line_number": ex.line_number,
            "code_line": ex.code_line,
        }

    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = min(self.max_seq_len, max((len(item["input_ids"]) for item in batch), default=1))

        input_ids = torch.full((len(batch), max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.zeros((len(batch),), dtype=torch.long)
        line_numbers = torch.zeros((len(batch),), dtype=torch.long)

        filenames: List[str] = []
        for i, item in enumerate(batch):
            ids = item["input_ids"][:max_len]
            mask = item["attention_mask"][:max_len]
            seq_len = len(ids)
            if seq_len > 0:
                input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
                attention_mask[i, :seq_len] = torch.tensor(mask, dtype=torch.long)
            labels[i] = int(item["label"])
            line_numbers[i] = int(item["line_number"])
            filenames.append(item["filename"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "filenames": filenames,
            "line_numbers": line_numbers,
        }


def create_dataloader(
    csv_path: str,
    tokenizer,
    batch_size: int,
    max_seq_len: int = 256,
    split: str = "all",
    shuffle: bool = True,
    num_workers: int = 0,
    drop_comment: bool = False,
    drop_blank: bool = False,
) -> DataLoader:
    dataset = CodeLineDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        split=split,
        drop_comment=drop_comment,
        drop_blank=drop_blank,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )


class FileSequenceDataset(Dataset):
    """
    Group lines by filename, then build one sample per file:
    - input_ids: [num_lines, max_tokens]
    - token_mask: [num_lines, max_tokens]
    - line_labels: [num_lines]
    - line_mask: [num_lines]
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_tokens_per_line: int = 64,
        max_lines_per_file: int = 256,
        line_window_size: Optional[int] = None,
        line_window_stride: Optional[int] = None,
        split: str = "all",
        drop_comment: bool = False,
        drop_blank: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_tokens_per_line = max_tokens_per_line
        self.max_lines_per_file = max_lines_per_file if max_lines_per_file > 0 else None
        self.line_window_size = line_window_size if (line_window_size is not None and line_window_size > 0) else None
        self.line_window_stride = line_window_stride if (line_window_stride is not None and line_window_stride > 0) else None
        self.pad_id = self._resolve_pad_id()

        examples = read_line_dp_csv(
            csv_path=csv_path,
            split=split,
            drop_comment=drop_comment,
            drop_blank=drop_blank,
        )

        grouped: Dict[str, List[CodeLineExample]] = {}
        for ex in examples:
            grouped.setdefault(ex.filename, []).append(ex)

        self.files: List[Dict] = []
        for filename, lines in grouped.items():
            lines_sorted = sorted(lines, key=lambda item: item.line_number)
            if self.line_window_size is None:
                self.files.append({"filename": filename, "lines": lines_sorted})
                continue

            window_size = self.line_window_size
            stride = self.line_window_stride if self.line_window_stride is not None else window_size
            stride = max(1, stride)

            total = len(lines_sorted)
            if total <= window_size:
                self.files.append({"filename": filename, "lines": lines_sorted})
                continue

            start_positions: List[int] = list(range(0, total - window_size + 1, stride))
            last_start = total - window_size
            if not start_positions or start_positions[-1] != last_start:
                start_positions.append(last_start)

            for start in start_positions:
                end = start + window_size
                self.files.append(
                    {
                        "filename": filename,
                        "lines": lines_sorted[start:end],
                    }
                )

    def _resolve_pad_id(self) -> int:
        for token in ("<pad>", "[PAD]"):
            idx = self.tokenizer.token_to_id(token)
            if idx is not None:
                return idx
        return 0

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        item = self.files[idx]
        filename = item["filename"]
        if self.max_lines_per_file is None:
            lines: List[CodeLineExample] = item["lines"]
        else:
            lines: List[CodeLineExample] = item["lines"][: self.max_lines_per_file]

        input_ids: List[List[int]] = []
        token_mask: List[List[int]] = []
        line_labels: List[int] = []
        line_numbers: List[int] = []

        for ex in lines:
            encoded = self.tokenizer.encode(ex.code_line)
            ids = encoded.ids[: self.max_tokens_per_line]
            mask = [1] * len(ids)
            input_ids.append(ids)
            token_mask.append(mask)
            line_labels.append(int(ex.label))
            line_numbers.append(int(ex.line_number))

        return {
            "filename": filename,
            "input_ids": input_ids,
            "token_mask": token_mask,
            "line_labels": line_labels,
            "line_numbers": line_numbers,
        }

    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        batch_max_lines = max((len(item["input_ids"]) for item in batch), default=1)
        if self.max_lines_per_file is None:
            max_lines = batch_max_lines
        else:
            max_lines = min(self.max_lines_per_file, batch_max_lines)

        max_tokens = min(
            self.max_tokens_per_line,
            max(
                (len(line_ids) for item in batch for line_ids in item["input_ids"]),
                default=1,
            ),
        )

        input_ids = torch.full((batch_size, max_lines, max_tokens), self.pad_id, dtype=torch.long)
        token_mask = torch.zeros((batch_size, max_lines, max_tokens), dtype=torch.long)
        line_labels = torch.zeros((batch_size, max_lines), dtype=torch.long)
        line_mask = torch.zeros((batch_size, max_lines), dtype=torch.long)
        line_numbers = torch.zeros((batch_size, max_lines), dtype=torch.long)

        filenames: List[str] = []
        for b, item in enumerate(batch):
            filenames.append(item["filename"])
            line_count = min(len(item["input_ids"]), max_lines)
            for l in range(line_count):
                ids = item["input_ids"][l][:max_tokens]
                mask = item["token_mask"][l][:max_tokens]
                token_len = len(ids)
                if token_len > 0:
                    input_ids[b, l, :token_len] = torch.tensor(ids, dtype=torch.long)
                    token_mask[b, l, :token_len] = torch.tensor(mask, dtype=torch.long)
                line_labels[b, l] = int(item["line_labels"][l])
                line_numbers[b, l] = int(item["line_numbers"][l])
                line_mask[b, l] = 1

        return {
            "filenames": filenames,
            "input_ids": input_ids,
            "token_mask": token_mask,
            "line_labels": line_labels,
            "line_mask": line_mask,
            "line_numbers": line_numbers,
        }


def create_file_sequence_dataloader(
    csv_path: str,
    tokenizer,
    batch_size: int,
    max_tokens_per_line: int = 64,
    max_lines_per_file: int = 256,
    split: str = "all",
    shuffle: bool = True,
    num_workers: int = 0,
    drop_comment: bool = False,
    drop_blank: bool = False,
) -> DataLoader:
    dataset = FileSequenceDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        max_tokens_per_line=max_tokens_per_line,
        max_lines_per_file=max_lines_per_file,
        split=split,
        drop_comment=drop_comment,
        drop_blank=drop_blank,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )
