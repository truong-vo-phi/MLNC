"""Bỏ hai lần tải thừa khi chạy KD: BEATs gốc (khi đã có bản fine-tune) và PANNs Cnn10 (khi nối tiếp từ resume.pt)."""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {text.count(old)}")
    return text.replace(old, new)


# 1) run_reviewkd.build_student: cho phép bỏ qua nạp PANNs
path = ROOT / "scripts" / "run_reviewkd.py"
s = path.read_text(encoding="utf-8")
s = replace_once(
    s,
    "def build_student(raw: dict[str, Any]) -> BaselineModel:\n"
    "    student = BaselineModel(BaselineConfig.from_dict(raw[\"model\"]), raw[\"frontend\"])\n"
    "    if student.config.pretrained:\n",
    "def build_student(raw: dict[str, Any], load_pretrained: bool = True) -> BaselineModel:\n"
    '    """load_pretrained=False: không tải/nạp PANNs (khi trọng số sẽ được ghi đè bởi resume.pt)."""\n'
    "    student = BaselineModel(BaselineConfig.from_dict(raw[\"model\"]), raw[\"frontend\"])\n"
    "    if student.config.pretrained and load_pretrained:\n",
    "build_student",
)
path.write_text(s, encoding="utf-8")

# 2) run_kd_beat
path = ROOT / "scripts" / "run_kd_beat.py"
z = path.read_text(encoding="utf-8")
z = replace_once(
    z,
    "    beats = BEATClassifier.from_yaml(BEATS_CONFIG)\n    if pretrain:\n        beats.load_pretrained(ensure_beats_checkpoint(raw))\n    finetuned = raw[\"teacher\"].get(\"beats_finetuned\")\n    if finetuned:\n",
    "    beats = BEATClassifier.from_yaml(BEATS_CONFIG)\n    finetuned = raw[\"teacher\"].get(\"beats_finetuned\")\n"
    "    if pretrain and not finetuned:  # bản fine-tune ghi đè toàn bộ trọng số (load_state_dict strict), không cần tải BEATs gốc\n"
    "        beats.load_pretrained(ensure_beats_checkpoint(raw))\n    if finetuned:\n",
    "teacher",
)
z = replace_once(
    z,
    "def build_frcrn_student(raw: dict[str, Any], use_frcrn: bool, frcrn_checkpoint: Path | None = None) -> nn.Module:\n    student = build_student(raw)\n",
    "def build_frcrn_student(\n    raw: dict[str, Any], use_frcrn: bool, frcrn_checkpoint: Path | None = None, load_pretrained: bool = True\n) -> nn.Module:\n    student = build_student(raw, load_pretrained)\n",
    "student",
)
z = replace_once(
    z,
    "    model = build_frcrn_student(raw, use_frcrn, frcrn_checkpoint).to(device)\n",
    "    resuming = resume_path is not None and resume_path.is_file()  # resume.pt ghi đè trọng số student nên không cần nạp PANNs\n"
    "    model = build_frcrn_student(raw, use_frcrn, frcrn_checkpoint, load_pretrained=not resuming).to(device)\n",
    "run_experiment",
)
path.write_text(z, encoding="utf-8")
print("patched")
