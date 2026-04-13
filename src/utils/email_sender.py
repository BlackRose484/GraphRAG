"""
Email utility — gửi kết quả experiment qua Gmail SMTP.

Sử dụng App Password (không phải mật khẩu Gmail thường).
Cấu hình qua .env:
    EMAIL_USER         Gmail gửi (vd: you@gmail.com)
    EMAIL_PASS         App Password 16 ký tự từ Google Account → Security
    ADMIN_EMAIL        Gmail nhận kết quả
"""
from __future__ import annotations

import json
import smtplib
from datetime import datetime
from email.message import EmailMessage
from typing import Any

from src.utils.logger import get_logger

_logger = get_logger(__name__)

_SMTP_HOST = "smtp.gmail.com"
_SMTP_PORT = 465   # SSL


def send_experiment_result(
    user: str,
    data: dict[str, Any],
    sender: str,
    app_password: str,
    receiver: str,
) -> None:
    """Gửi kết quả experiment dưới dạng email + file JSON đính kèm.

    Args:
        user:         Tên người tham gia.
        data:         Dict kết quả (cùng cấu trúc với file JSON lưu trên disk).
        sender:       Địa chỉ Gmail gửi (EMAIL_USER).
        app_password: App Password 16 ký tự (EMAIL_PASS).
        receiver:     Địa chỉ Gmail nhận (ADMIN_EMAIL).

    Raises:
        RuntimeError: Nếu gửi thất bại.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    steps = data.get("steps", [])

    # ── Body text ──────────────────────────────────────────────────────────────
    lines: list[str] = [
        f"Kết quả thử nghiệm từ người dùng: {user}",
        f"Thời điểm nộp: {timestamp}",
        f"Số câu hoàn thành: {data.get('total_completed', 0)}/4",
        "",
        "─" * 50,
    ]
    for step in steps:
        rat  = step.get("rating", {})
        ps   = rat.get("per_system", {})
        lines.append(f"\nBước {step['step']} — {step.get('category', '')}")
        lines.append(f"Câu hỏi : {step.get('question', '')}")
        lines.append(f"Case ID : {step.get('case_id', '')}")
        for sk, label in [("graphrag", "GraphRAG"), ("rag", "RAG"), ("chat", "LLM")]:
            s = ps.get(sk, {})
            if s:
                lines.append(
                    f"  {label:8s}: Chính xác={s.get('accuracy','-')} | "
                    f"Đầy đủ={s.get('completeness','-')} | "
                    f"Tự nhiên={s.get('naturalness','-')}"
                )
        lines.append(f"  Tốt nhất : {rat.get('best_system', '?')}")
        if rat.get("note"):
            lines.append(f"  Ghi chú  : {rat['note']}")
    lines.append("\n" + "─" * 50)
    body = "\n".join(lines)

    # ── Attachment ─────────────────────────────────────────────────────────────
    json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    filename   = f"experiment_{user}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

    # ── Compose ────────────────────────────────────────────────────────────────
    msg = EmailMessage()
    msg["Subject"] = f"[Experiment] {user} — {timestamp}"
    msg["From"]    = sender
    msg["To"]      = receiver
    msg.set_content(body, charset="utf-8")
    msg.add_attachment(
        json_bytes,
        maintype="application",
        subtype="json",
        filename=filename,
    )

    # ── Send ───────────────────────────────────────────────────────────────────
    try:
        with smtplib.SMTP_SSL(_SMTP_HOST, _SMTP_PORT) as smtp:
            smtp.login(sender, app_password)
            smtp.send_message(msg)
        _logger.info(
            "Experiment result sent → %s  (user=%s)", receiver, user
        )
    except smtplib.SMTPAuthenticationError as exc:
        _logger.error("Gmail auth failed: %s", exc)
        raise RuntimeError(
            "Xác thực Gmail thất bại. Kiểm tra lại EMAIL_USER và EMAIL_PASS (App Password)."
        ) from exc
    except smtplib.SMTPException as exc:
        _logger.error("SMTP error: %s", exc)
        raise RuntimeError(f"Gửi email thất bại: {exc}") from exc
