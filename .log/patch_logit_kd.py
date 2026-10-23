"""Một lần: thêm KD logit (softmax, nhiệt độ) vào ReviewKDTrainer cùng teacher NoiseTeacher và CLI run_kd_beat."""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {text.count(old)}")
    return text.replace(old, new)


# 1) config
config_path = ROOT / "src" / "train" / "config.py"
s = config_path.read_text(encoding="utf-8")
s = replace_once(
    s,
    "    review_warmup_epochs: int = 5\n    hcl_levels: tuple[int, ...] = (4, 2, 1)\n",
    "    review_warmup_epochs: int = 5\n    hcl_levels: tuple[int, ...] = (4, 2, 1)\n"
    "    logit_weight: float = 0.0  # KD trên logit teacher (softmax, nhiệt độ); 0 = tắt\n"
    "    logit_temperature: float = 4.0\n",
    "config fields",
)
s = replace_once(
    s,
    '        if self.review_warmup_epochs < 0:\n            raise ValueError(f"review_warmup_epochs must be >= 0, got {self.review_warmup_epochs}")\n',
    '        if self.review_warmup_epochs < 0:\n            raise ValueError(f"review_warmup_epochs must be >= 0, got {self.review_warmup_epochs}")\n'
    '        if self.logit_weight < 0:\n            raise ValueError(f"logit_weight must be >= 0, got {self.logit_weight}")\n'
    '        if self.logit_temperature <= 0:\n            raise ValueError(f"logit_temperature must be > 0, got {self.logit_temperature}")\n',
    "config validation",
)
config_path.write_text(s, encoding="utf-8")

# 2) trainer
trainer_path = ROOT / "src" / "train" / "reviewkd_trainer.py"
t = trainer_path.read_text(encoding="utf-8")
t = replace_once(t, "import numpy as np\nimport torch\nfrom torch import Tensor, nn\n", "import numpy as np\nimport torch\nimport torch.nn.functional as F\nfrom torch import Tensor, nn\n", "imports")
t = replace_once(t, "        self._resumed_stale = 0\n", "        self._resumed_stale = 0\n        self._teacher_logits: Tensor | None = None\n", "init attr")
t = replace_once(
    t,
    "        return self.config.review_weight * scale\n\n    def _reset_stats",
    "        return self.config.review_weight * scale\n\n"
    "    def logit_weight(self, epoch: int) -> float:\n"
    "        warmup = self.config.review_warmup_epochs\n"
    "        scale = 1.0 if warmup == 0 else min(1.0, (epoch - 1) / warmup)\n"
    "        return self.config.logit_weight * scale\n\n    def _reset_stats",
    "logit_weight()",
)
t = replace_once(t, "        self._review_sum = 0.0\n        self._level_sum", "        self._review_sum = 0.0\n        self._logit_sum = 0.0\n        self._level_sum", "reset stats")
t = replace_once(t, "        features = self.teacher(waveform)\n", "        features = self._call_teacher(waveform)\n", "call teacher")
t = replace_once(
    t,
    "    def _teacher_features(self, waveform: Tensor) -> Tensor:",
    "    def _call_teacher(self, waveform: Tensor) -> Tensor:\n"
    '        """Gọi teacher lấy lưới đặc trưng. Teacher có logit thì đặt `self._teacher_logits` ở đây cho KD logit."""\n'
    "        return self.teacher(waveform)\n\n    def _teacher_features(self, waveform: Tensor) -> Tensor:",
    "_call_teacher",
)
t = replace_once(
    t,
    "        weight = self.review_weight(self._epoch)\n        use_review = self.teacher is not None and weight > 0\n"
    "        output = self.model(waveform, return_features=use_review)\n        ce = self.criterion(output.logits, labels)\n        loss = ce\n"
    "        if use_review:\n            teacher_features = self._teacher_features(waveform)\n",
    "        weight = self.review_weight(self._epoch)\n        logit_weight = self.logit_weight(self._epoch)\n"
    "        use_review = self.teacher is not None and weight > 0\n        use_logit = self.teacher is not None and logit_weight > 0\n"
    "        output = self.model(waveform, return_features=use_review)\n        ce = self.criterion(output.logits, labels)\n        loss = ce\n"
    "        teacher_features = self._teacher_features(waveform) if (use_review or use_logit) else None\n"
    "        if use_review:\n",
    "training_step head",
)
t = replace_once(
    t,
    "        self._ce_sum += ce.item() * len(labels)\n        self._seen += len(labels)\n",
    "        if use_logit:\n"
    "            if self._teacher_logits is None:\n"
    '                raise ValueError("logit_weight > 0 cần teacher trả logit (NoiseTeacher.forward(..., with_logits=True))")\n'
    "            temperature = self.config.logit_temperature\n"
    "            logit_kd = (\n"
    "                F.kl_div(\n"
    "                    F.log_softmax(output.logits / temperature, dim=-1),\n"
    "                    F.softmax(self._teacher_logits.detach().float() / temperature, dim=-1),\n"
    '                    reduction="batchmean",\n'
    "                )\n"
    "                * temperature**2\n"
    "            )\n"
    "            loss = loss + logit_weight * logit_kd\n"
    "            self._logit_sum += logit_kd.item() * len(labels)\n"
    "        self._ce_sum += ce.item() * len(labels)\n        self._seen += len(labels)\n",
    "logit loss",
)
t = replace_once(
    t,
    '            "review_weight": self.review_weight(self._epoch),\n        }\n',
    '            "review_weight": self.review_weight(self._epoch),\n        }\n'
    "        if self.config.logit_weight > 0:\n"
    '            summary["logit_kd"] = self._logit_sum / seen\n'
    '            summary["logit_weight"] = self.logit_weight(self._epoch)\n',
    "summary",
)
trainer_path.write_text(t, encoding="utf-8")

# 3) run_kd_beat
run_path = ROOT / "scripts" / "run_kd_beat.py"
z = run_path.read_text(encoding="utf-8")
z = replace_once(
    z,
    "    @torch.no_grad()\n    def forward(self, noise: Tensor) -> Tensor:\n"
    "        output = self.beats(rms_normalize(noise, self.target_rms), layers=self.layers)\n"
    "        return torch.stack([tokens_to_grid(hidden, self.freq_bins) for hidden in output.hidden])",
    "    @torch.no_grad()\n    def forward(self, noise: Tensor, with_logits: bool = False) -> Tensor | tuple[Tensor, Tensor]:\n"
    "        output = self.beats(rms_normalize(noise, self.target_rms), layers=self.layers)\n"
    "        grids = torch.stack([tokens_to_grid(hidden, self.freq_bins) for hidden in output.hidden])\n"
    "        return (grids, output.logits) if with_logits else grids",
    "teacher forward",
)
z = replace_once(
    z,
    "    def _teacher_features(self, waveform):\n        return super()._teacher_features(self._noise)",
    "    def _call_teacher(self, waveform):\n"
    "        if self.config.logit_weight > 0:\n"
    "            grids, self._teacher_logits = self.teacher(self._noise, with_logits=True)\n"
    "            return grids\n"
    "        return self.teacher(self._noise)",
    "trainer teacher call",
)
z = replace_once(
    z,
    "    review_weight: float | None = None,\n    smoke: bool = False,\n",
    "    review_weight: float | None = None,\n    logit_weight: float | None = None,\n    logit_temperature: float | None = None,\n    smoke: bool = False,\n",
    "run_experiment args",
)
z = replace_once(
    z,
    'overrides = {k: v for k, v in (("review_weight", review_weight), ("epochs", epochs), ("patience", patience)) if v is not None}',
    "overrides = {\n        k: v\n        for k, v in (\n"
    '            ("review_weight", review_weight),\n            ("logit_weight", logit_weight),\n            ("logit_temperature", logit_temperature),\n'
    '            ("epochs", epochs),\n            ("patience", patience),\n        )\n        if v is not None\n    }',
    "overrides",
)
z = replace_once(
    z,
    "def experiment_name(arch: str, use_frcrn: bool, review_weight: float | None, pretrain_teacher: bool, finetuned_teacher: bool = False) -> str:",
    "def experiment_name(\n    arch: str,\n    use_frcrn: bool,\n    review_weight: float | None,\n    pretrain_teacher: bool,\n    finetuned_teacher: bool = False,\n    logit_kd: bool = False,\n) -> str:",
    "experiment_name sig",
)
z = replace_once(
    z,
    '        "ft" if finetuned_teacher else ("pre" if pretrain_teacher else "rand"),\n    ])',
    '        "ft" if finetuned_teacher else ("pre" if pretrain_teacher else "rand"),\n        *(["logit"] if logit_kd else []),\n    ])',
    "experiment_name body",
)
z = replace_once(
    z,
    '    parser.add_argument("--epochs", type=int, default=None)\n',
    '    parser.add_argument("--logit-weight", type=float, default=None, help="trọng số KD logit (0 = tắt); cần teacher đã fine-tune 36 lớp")\n'
    '    parser.add_argument("--logit-temperature", type=float, default=None)\n    parser.add_argument("--epochs", type=int, default=None)\n',
    "cli",
)
z = replace_once(
    z,
    'args.review_weight, pretrain, bool(raw["teacher"].get("beats_finetuned")))',
    'args.review_weight, pretrain, bool(raw["teacher"].get("beats_finetuned")), bool(args.logit_weight))',
    "name call",
)
z = replace_once(
    z,
    "review_weight=args.review_weight, smoke=args.smoke,",
    "review_weight=args.review_weight, logit_weight=args.logit_weight, logit_temperature=args.logit_temperature, smoke=args.smoke,",
    "run call",
)
run_path.write_text(z, encoding="utf-8")
print("patched config, trainer, run_kd_beat")
