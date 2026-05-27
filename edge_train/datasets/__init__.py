"""Built-in datasets and modality inference."""

from pathlib import Path
from typing import Any


def _urgency_dataset() -> dict[str, Any]:
    lines = [
        ("服务器挂了快来看看", "紧急"),
        ("我发烧了在医院", "紧急"),
        ("生产线停机了", "紧急"),
        ("数据库连接失败", "紧急"),
        ("客户投诉升级", "紧急"),
        ("明天下午三点开会", "重要"),
        ("这个方案需要你确认", "重要"),
        ("周报请今天提交", "重要"),
        ("项目进度更新", "重要"),
        ("Q2预算需要审批", "重要"),
        ("快递到了放门口", "一般"),
        ("今晚吃什么", "一般"),
        ("新来了个同事", "一般"),
        ("WiFi密码是多少", "一般"),
        ("打印机卡纸了", "一般"),
        ("双十一大促开始了", "可忽略"),
        ("拼多多帮我砍一刀", "可忽略"),
        ("会员积分即将过期", "可忽略"),
        ("免费领取保险", "可忽略"),
        ("你中奖了！", "可忽略"),
    ]
    # Repeat to reach ~400 samples
    multiplier = 20
    csv_lines = ["text,urgency"]
    for _ in range(multiplier):
        for text, label in lines:
            escaped = text.replace('"', '""')
            csv_lines.append(f'"{escaped}",{label}')
    return {
        "modality": "text",
        "samples": len(lines) * multiplier,
        "classes": ["紧急", "重要", "一般", "可忽略"],
        "description": "消息紧急程度分类",
        "csv_content": "\n".join(csv_lines),
    }


def _expense_dataset() -> dict[str, Any]:
    lines = [
        ("美团外卖 35元", "餐饮"),
        ("星巴克中杯拿铁", "餐饮"),
        ("海底捞晚餐 268元", "餐饮"),
        ("麦当劳套餐", "餐饮"),
        ("盒马鲜生 水果", "餐饮"),
        ("滴滴出行 12元", "交通"),
        ("地铁-五道口", "交通"),
        ("高铁票 北京-上海", "交通"),
        ("加油 300元", "交通"),
        ("停车费 15元", "交通"),
        ("京东 纸巾*2", "购物"),
        ("拼多多 数据线", "购物"),
        ("小米商城 充电宝", "购物"),
        ("淘宝 衣服", "购物"),
        ("无印良品 收纳盒", "购物"),
        ("电费缴费 200元", "居住"),
        ("房租 5000元", "居住"),
        ("水费 80元", "居住"),
        ("物业费 300元", "居住"),
        ("燃气费 50元", "居住"),
        ("Steam 游戏", "娱乐"),
        ("电影票 两张", "娱乐"),
        ("网易云音乐会员", "娱乐"),
        ("羽毛球场地费", "娱乐"),
        ("Netflix 订阅", "娱乐"),
    ]
    multiplier = 20
    csv_lines = ["text,category"]
    for _ in range(multiplier):
        for text, label in lines:
            escaped = text.replace('"', '""')
            csv_lines.append(f'"{escaped}",{label}')
    return {
        "modality": "text",
        "samples": len(lines) * multiplier,
        "classes": ["餐饮", "交通", "购物", "居住", "娱乐"],
        "description": "个人消费意图分类",
        "csv_content": "\n".join(csv_lines),
    }


_BUILTIN_DATASETS: dict[str, dict[str, Any]] = {}


def _ensure_loaded() -> None:
    if not _BUILTIN_DATASETS:
        _BUILTIN_DATASETS["urgent"] = _urgency_dataset()
        _BUILTIN_DATASETS["expense"] = _expense_dataset()


def get_builtin() -> dict[str, dict[str, Any]]:
    _ensure_loaded()
    return dict(_BUILTIN_DATASETS)


def list_builtin() -> dict[str, dict[str, Any]]:
    return get_builtin()


def infer_modality(path: Path) -> str:
    """Infer data modality from a file path."""
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        return _infer_modality_from_csv(path)
    if suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        return "image"
    if suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"):
        return "video"
    if suffix in (".wav", ".mp3", ".flac", ".ogg"):
        return "sound"
    return "unknown"


def infer_modality_from_path(path: str) -> str:
    _ensure_loaded()
    if path.startswith("builtin:"):
        name = path.split(":", 1)[1]
        builtin = _BUILTIN_DATASETS.get(name)
        if builtin:
            return builtin.get("modality", "text")
        return "unknown"
    if not Path(path).exists() and path in _BUILTIN_DATASETS:
        return _BUILTIN_DATASETS[path].get("modality", "text")
    return infer_modality(Path(path))


def resolve_dataset_path(path: str) -> tuple[str, str | None]:
    """Materialize built-in datasets to a temp CSV. Returns (csv_path, modality|None).

    Accepts ``builtin:urgent`` or bare ``urgent`` / ``expense`` when not a local file.
    """
    _ensure_loaded()
    name: str | None = None
    if path.startswith("builtin:"):
        name = path.split(":", 1)[1]
    elif not Path(path).exists() and path in _BUILTIN_DATASETS:
        name = path

    if name is None:
        return path, None

    builtin = _BUILTIN_DATASETS.get(name)
    if not builtin:
        available = ", ".join(sorted(_BUILTIN_DATASETS))
        raise ValueError(
            f"Unknown built-in dataset '{name}'. Available: {available or '(none)'}"
        )

    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="coralflow-")) / f"{name}.csv"
    tmp.write_text(builtin["csv_content"], encoding="utf-8")
    return str(tmp), builtin.get("modality")


def _infer_modality_from_csv(path: Path) -> str:
    """Heuristic: CSV with 1 text column = text, mostly numeric = table."""
    import csv

    with open(path) as f:
        reader = csv.DictReader(f)
        sample = []
        for i, row in enumerate(reader):
            if i >= 20:
                break
            sample.append(row)

    if not sample:
        return "text"

    headers = [h for h in (reader.fieldnames or sample[0].keys()) if h.strip()]
    text_keywords = {
        "text",
        "message",
        "content",
        "sentence",
        "review",
        "comment",
        "description",
    }
    text_headers = [h for h in headers if h.lower() in text_keywords]

    if text_headers:
        return "text"

    # Check if most columns are numeric
    numeric_count = 0
    for row in sample:
        for h in headers:
            val = row.get(h, "")
            try:
                float(val)
                numeric_count += 1
            except ValueError:
                pass

    total_cells = len(sample) * len(headers)
    if numeric_count > total_cells * 0.7:
        return "table"

    return "text"
