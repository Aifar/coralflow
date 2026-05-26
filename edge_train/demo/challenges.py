"""Challenge samples for retrain-loop demos (not in builtin training templates)."""

from __future__ import annotations

import csv
from pathlib import Path

# (text, ground_truth) — paraphrases absent from builtin urgent templates.
URGENT_DRIFT_CASES: list[tuple[str, str]] = [
    ("生产停线了速来处理", "紧急"),
    ("数据库宕机了赶紧看", "紧急"),
    ("人在医院发烧了", "紧急"),
    ("客户威胁要投诉升级", "紧急"),
    ("系统全面崩溃请支援", "紧急"),
    ("下午三点会议别忘了", "重要"),
    ("这份合同请你今天确认", "重要"),
    ("项目周报务必今晚提交", "重要"),
    ("预算审批麻烦看一下", "重要"),
    ("快递放在门口了", "一般"),
    ("晚上一起吃饭吗", "一般"),
    ("双十一大促又来了", "可忽略"),
    ("点击链接免费领取保险", "可忽略"),
    ("拼多多砍一刀帮帮忙", "可忽略"),
]

# Labels withheld from baseline training in --hard mode (model has not seen these).
URGENT_HARD_TRAIN_LABELS = ("一般", "可忽略")


def write_urgent_challenge_csv(path: Path) -> Path:
    """Write challenge rows to CSV (text, urgency)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "urgency"])
        for text, label in URGENT_DRIFT_CASES:
            writer.writerow([text, label])
    return path
