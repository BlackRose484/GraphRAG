"""
Email fetcher — kéo các email kết quả thử nghiệm / đánh giá ưu tiên về máy.

Cùng cấu hình với :mod:`src.utils.email_sender`:
    EMAIL_USER         Gmail nhận (= ADMIN_EMAIL trong luồng gửi hiện tại)
    EMAIL_PASS         App Password 16 ký tự (dùng được cho cả SMTP lẫn IMAP)

Phân loại email dựa vào dòng "Số câu hoàn thành: X/4" trong body do
``email_sender.send_experiment_result`` sinh ra:

    X = 4  → experiment  (thử nghiệm)
    X = 0  → preference  (đánh giá ưu tiên — data dict thiếu key total_completed)

Không dùng package bên ngoài: chỉ stdlib (imaplib + email).
"""
from __future__ import annotations

import email
import imaplib
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from email.header import decode_header, make_header
from email.message import Message
from email.utils import parsedate_to_datetime
from typing import Iterator, Literal, Optional

from src.utils.logger import get_logger

_logger = get_logger(__name__)

_IMAP_HOST = "imap.gmail.com"
_IMAP_PORT = 993

EmailKind = Literal["experiment", "preference", "unknown"]

_COMPLETION_RE = re.compile(r"Số câu hoàn thành:\s*(\d+)\s*/\s*4")


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class ParsedEmail:
    """Một email đã được parse: metadata + body + attachment JSON."""

    message_id: str
    subject: str
    sender: str
    date: Optional[datetime]
    body: str
    kind: EmailKind
    attachment_name: Optional[str]
    attachment_bytes: Optional[bytes]

    @property
    def has_json_attachment(self) -> bool:
        return bool(self.attachment_name and self.attachment_bytes)


# ── Public helpers ────────────────────────────────────────────────────────────

def classify(body: str) -> EmailKind:
    """Phân loại email dựa vào dòng "Số câu hoàn thành: X/4" trong body."""
    m = _COMPLETION_RE.search(body)
    if not m:
        return "unknown"
    n = int(m.group(1))
    if n == 4:
        return "experiment"
    if n == 0:
        return "preference"
    return "unknown"


# ── Fetcher ───────────────────────────────────────────────────────────────────

class ExperimentEmailFetcher:
    """Connect tới Gmail qua IMAP, lấy các email khớp subject filter.

    Dùng như context manager để tự động logout::

        with ExperimentEmailFetcher(user, app_password) as fetcher:
            for mail in fetcher.fetch():
                ...
    """

    def __init__(
        self,
        user: str,
        app_password: str,
        host: str = _IMAP_HOST,
        port: int = _IMAP_PORT,
    ) -> None:
        self._user = user
        self._password = app_password
        self._host = host
        self._port = port
        self._conn: Optional[imaplib.IMAP4_SSL] = None

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self) -> "ExperimentEmailFetcher":
        self._conn = imaplib.IMAP4_SSL(self._host, self._port)
        try:
            self._conn.login(self._user, self._password)
        except imaplib.IMAP4.error as exc:
            raise RuntimeError(
                "Đăng nhập Gmail IMAP thất bại. Kiểm tra EMAIL_USER / EMAIL_PASS "
                "(App Password) và đảm bảo IMAP đã bật trong Gmail Settings."
            ) from exc
        _logger.info("IMAP connected: %s@%s", self._user, self._host)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._conn is None:
            return
        try:
            try:
                self._conn.close()
            except imaplib.IMAP4.error:
                pass
            self._conn.logout()
        finally:
            self._conn = None

    # ── Fetch ──────────────────────────────────────────────────────────────

    def fetch(
        self,
        subject: str = "[Experiment]",
        mailbox: str = "INBOX",
        since: Optional[datetime] = None,
    ) -> Iterator[ParsedEmail]:
        """Stream các email khớp filter.

        Args:
            subject:  chuỗi xuất hiện trong Subject (mặc định ``[Experiment]``).
            mailbox:  hộp thư cần SELECT.
            since:    chỉ lấy email từ ngày này trở đi (IMAP ``SINCE``).

        Yields:
            :class:`ParsedEmail` cho từng email tìm được.
        """
        if self._conn is None:
            raise RuntimeError("Fetcher chưa connect — dùng trong `with` block.")

        status, _ = self._conn.select(mailbox, readonly=True)
        if status != "OK":
            raise RuntimeError(f"Không SELECT được mailbox {mailbox!r}")

        criteria = [f'SUBJECT "{subject}"']
        if since is not None:
            criteria.append(f'SINCE "{since.strftime("%d-%b-%Y")}"')
        search_query = "(" + " ".join(criteria) + ")"

        status, data = self._conn.search(None, search_query)
        if status != "OK":
            raise RuntimeError(f"IMAP SEARCH lỗi: {data!r}")

        ids = data[0].split() if data and data[0] else []
        _logger.info("IMAP SEARCH %s → %d email khớp", search_query, len(ids))

        for num in ids:
            parsed = self._fetch_one(num)
            if parsed is not None:
                yield parsed

    def _fetch_one(self, num: bytes) -> Optional[ParsedEmail]:
        assert self._conn is not None
        status, data = self._conn.fetch(num, "(RFC822)")
        if status != "OK" or not data or not isinstance(data[0], tuple):
            _logger.warning("FETCH %s thất bại", num)
            return None

        raw = data[0][1]
        msg = email.message_from_bytes(raw)

        message_id = (msg.get("Message-ID") or "").strip()
        subject = _decode(msg.get("Subject", ""))
        sender = _decode(msg.get("From", ""))
        date = _parse_date(msg.get("Date"))
        body = _extract_text_body(msg)
        att_name, att_bytes = _extract_json_attachment(msg)

        return ParsedEmail(
            message_id=message_id,
            subject=subject,
            sender=sender,
            date=date,
            body=body,
            kind=classify(body),
            attachment_name=att_name,
            attachment_bytes=att_bytes,
        )


# ── Private helpers ───────────────────────────────────────────────────────────

def _decode(raw: str) -> str:
    if not raw:
        return ""
    try:
        return str(make_header(decode_header(raw)))
    except Exception:
        return raw


def _parse_date(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None


def _extract_text_body(msg: Message) -> str:
    """Lấy phần text/plain đầu tiên (không phải attachment) làm body."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() != "text/plain":
                continue
            disp = (part.get("Content-Disposition") or "").lower()
            if "attachment" in disp:
                continue
            return _payload_to_text(part)
        return ""
    if msg.get_content_type() == "text/plain":
        return _payload_to_text(msg)
    return ""


def _payload_to_text(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _extract_json_attachment(msg: Message) -> tuple[Optional[str], Optional[bytes]]:
    """Trả về (filename, bytes) của attachment JSON đầu tiên gặp được."""
    if not msg.is_multipart():
        return None, None
    for part in msg.walk():
        disp = (part.get("Content-Disposition") or "").lower()
        if "attachment" not in disp:
            continue
        filename = _decode(part.get_filename() or "")
        if not filename.lower().endswith(".json"):
            continue
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        return filename, payload
    return None, None
