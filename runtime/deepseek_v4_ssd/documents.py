"""Bounded, process-isolated document extraction. Never execute Office/PDF content.

OOXML is semantic extraction, not a layout renderer. PDF includes page rasters so
scanned pages are not silently dropped. Native parser memory is not a sandbox.
"""
from __future__ import annotations

import base64
from contextlib import closing
import hashlib
import io
import json
import math
import os
from pathlib import PurePosixPath
import posixpath
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
import zipfile

PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
DOCUMENT_TYPES = frozenset((PDF, DOCX, PPTX, XLSX))
MAX_DOCUMENTS = 4
MAX_PAGES = 4
MAX_TEXT_BYTES = 64 * 1024
MAX_ZIP_BYTES = 32 * 1024**2
MAX_OUTPUT_BYTES = 16 * 1024**2
MAX_WORKER_RSS = 768 * 1024**2
_WORKERS = threading.BoundedSemaphore(2)
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"


class DocumentError(ValueError):
    pass


class Output:
    def __init__(self, digest):
        self.parts = []
        self.digest = digest
        self.text_bytes = 0
        self.images = 0

    def text(self, location, text):
        if not text:
            return
        value = f"\n[Document {self.digest}; {location}]\n{text}\n"
        self.text_bytes += len(value.encode())
        if self.text_bytes > MAX_TEXT_BYTES:
            raise DocumentError("Document text exceeds 64 KiB; select a smaller document.")
        self.parts.append({"type": "text", "text": value})

    def image(self, location, data):
        from PIL import Image
        from .media import MAX_IMAGE_PIXELS, MAX_ASSET_BYTES
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ("PNG", "JPEG", "WEBP") or getattr(image, "n_frames", 1) != 1:
                raise DocumentError("Document contains an unsupported or animated image.")
            if image.width * image.height > MAX_IMAGE_PIXELS or len(data) > MAX_ASSET_BYTES:
                raise DocumentError("Document image exceeds the image budget.")
            image.load()
            mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[image.format]
        self.images += 1
        if self.images > 8:
            raise DocumentError("Document contains more than eight images.")
        self.text(location, "[Image follows]")
        self.parts.append({"type": "image", "mime": mime, "data": base64.b64encode(data).decode()})


class Package:
    def __init__(self, data):
        self.zip = zipfile.ZipFile(io.BytesIO(data))
        entries = self.zip.infolist()
        if len(entries) > 512 or sum(i.file_size for i in entries) > MAX_ZIP_BYTES:
            raise DocumentError("Office package exceeds 512 entries or 32 MiB expanded bytes.")
        names = [i.filename for i in entries]
        if len(set(names)) != len(names):
            raise DocumentError("Duplicate Office package entries are not allowed.")
        for entry in entries:
            name = entry.filename
            if (PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or "\\" in name
                    or len(name) > 512 or any(ord(c) < 32 for c in name) or entry.flag_bits & 1
                    or (entry.external_attr >> 16) & 0o170000 == 0o120000):
                raise DocumentError("Unsafe Office package entry.")
            lower = name.lower()
            if any(s in lower for s in ("vbaproject", "/activex/", "/embeddings/")):
                raise DocumentError("Macros, ActiveX and embedded objects are not supported.")
        self.names = set(names)
        self.bytes_read = 0
        self._xml = {}
        self._relations = {}
        types = self.xml("[Content_Types].xml")
        if types.tag != "{http://schemas.openxmlformats.org/package/2006/content-types}Types":
            raise DocumentError("Invalid Office content types.")
        if any("macroenabled" in node.get("ContentType", "").lower() for node in types):
            raise DocumentError("Macro-enabled Office documents are not supported.")

    def read(self, name):
        if name not in self.names:
            raise DocumentError("Office package references a missing part.")
        info = self.zip.getinfo(name)
        self.bytes_read += info.file_size
        if self.bytes_read > 2 * MAX_ZIP_BYTES:
            raise DocumentError("Office package expansion budget exceeded.")
        with self.zip.open(info) as stream:
            data = stream.read(MAX_ZIP_BYTES + 1)
        if len(data) != info.file_size or len(data) > MAX_ZIP_BYTES:
            raise DocumentError("Office package size mismatch.")
        return data

    def xml(self, name):
        if name not in self._xml:
            data = self.read(name)
            # Normalize encoding before the DTD check, including UTF-16 OOXML.
            text = data.decode("utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
            if "\0" in text or "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
                raise DocumentError("XML DTDs and entities are disabled.")
            root = ET.fromstring(text)
            stack = [(root, 0)]
            count = 0
            while stack:
                node, depth = stack.pop()
                count += 1
                if depth > 64 or count > 100_000:
                    raise DocumentError("Office XML structure exceeds its limit.")
                local = node.tag.rsplit("}", 1)[-1]
                if local in ("altChunk", "OLEObject", "chart", "relIds", "AlternateContent", "oMath", "oMathPara",
                             "audioFile", "videoFile", "wavAudioFile", "audio", "video", "media", "contentPart"):
                    raise DocumentError("Document contains unsupported embedded content or alternate drawings.")
                stack.extend((child, depth + 1) for child in node)
            self._xml[name] = root
        return self._xml[name]

    def relations(self, source):
        if source not in self._relations:
            path = posixpath.join(posixpath.dirname(source), "_rels", posixpath.basename(source) + ".rels")
            result = {}
            if path in self.names:
                for rel in self.xml(path):
                    if rel.tag != REL + "Relationship" or not rel.get("Id") or rel.get("Id") in result:
                        raise DocumentError("Invalid Office relationship.")
                    result[rel.get("Id")] = rel.attrib
            self._relations[source] = result
        return self._relations[source]

    def target(self, source, key):
        rel = self.relations(source).get(key)
        if rel is None or rel.get("TargetMode", "Internal") != "Internal":
            raise DocumentError("External or missing content references are not supported.")
        value = unquote(rel.get("Target", ""))
        if not value or urlsplit(value).scheme or urlsplit(value).netloc or "\\" in value:
            raise DocumentError("Invalid Office content reference.")
        target = posixpath.normpath(value.lstrip("/") if value.startswith("/") else posixpath.join(posixpath.dirname(source), value))
        if target.startswith("../") or target not in self.names:
            raise DocumentError("Office content reference escapes the package or is missing.")
        return target

    def content(self, source, node, out, location):
        pending = []
        def flush():
            out.text(location, "".join(pending))
            pending.clear()
        for item in node.iter():
            if item.tag in (W + "t", A + "t"):
                pending.append(item.text or "")
            elif item.tag in (W + "tab",):
                pending.append("\t")
            elif item.tag in (W + "br", A + "br", W + "p", A + "p"):
                pending.append("\n")
            elif item.tag in (A + "blip", "{urn:schemas-microsoft-com:vml}imagedata"):
                flush()
                key = item.get(R + "embed") or item.get(R + "link") or item.get(R + "id")
                out.image(location, self.read(self.target(source, key)))
        flush()


def _pdf(data, out):
    import pypdfium2 as pdfium
    with pdfium.PdfDocument(data) as document:
        if not 1 <= len(document) <= MAX_PAGES:
            raise DocumentError("PDF exceeds four pages; select a smaller document.")
        if pdfium.raw.FPDF_GetFormType(document) != 0:
            raise DocumentError("Interactive PDF forms are not supported; export a flat PDF first.")
        out.text("extraction", "PDF page text and page images follow. Scripts and embedded attachments are not imported.")
        for index in range(len(document)):
            with closing(document[index]) as page:
                width, height = page.get_size()
                if not all(math.isfinite(v) and 0 < v <= 14400 for v in (width, height)):
                    raise DocumentError("Invalid or oversized PDF page geometry.")
                with closing(page.get_textpage()) as text:
                    if text.count_chars() > MAX_TEXT_BYTES:
                        raise DocumentError("PDF text exceeds its limit.")
                    out.text(f"page {index + 1}", text.get_text_range())
                # Fixed upper bound, including scanned pages. No native-resolution
                # raster allocation followed by a late resize.
                with closing(page.render(scale=min(1.5, 768 / max(width, height)), may_draw_forms=False)) as bitmap:
                    image = bitmap.to_pil()
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    out.image(f"page {index + 1}", buffer.getvalue())


def _office(data, mime, out):
    package = Package(data)
    out.text("extraction", "Office stored text, tables and raster images follow; page layout and formatting are not rendered. External hyperlinks are not fetched; formulas are not executed.")
    if mime == DOCX:
        source = "word/document.xml"
        root = package.xml(source)
        if root.tag != W + "document":
            raise DocumentError("Not a supported DOCX document.")
        body = root.find(W + "body")
        if body is None:
            raise DocumentError("DOCX body is missing.")
        for index, node in enumerate(body, 1):
            if node.tag == W + "tbl":
                for row_index, row in enumerate(node.findall(W + "tr"), 1):
                    for col_index, cell in enumerate(row.findall(W + "tc"), 1):
                        package.content(source, cell, out, f"body block {index}; table row {row_index}, cell {col_index}")
            else:
                package.content(source, node, out, f"body block {index}")
        # Include headers, footers and notes rather than silently dropping them.
        extras = {name for name in package.names if name in ("word/footnotes.xml", "word/endnotes.xml", "word/comments.xml")}
        for rel in package.relations(source).values():
            if rel.get("Type", "").rsplit("/", 1)[-1] in ("header", "footer"):
                extras.add(package.target(source, rel["Id"]))
        for name in sorted(extras):
            package.content(name, package.xml(name), out, name)
    elif mime == PPTX:
        source = "ppt/presentation.xml"
        root = package.xml(source)
        if root.tag != P + "presentation":
            raise DocumentError("Not a supported PPTX presentation.")
        slides = root.findall(f"{P}sldIdLst/{P}sldId")
        if not 1 <= len(slides) <= MAX_PAGES:
            raise DocumentError("Presentation exceeds four slides or is empty.")
        for index, slide in enumerate(slides, 1):
            target = package.target(source, slide.get(R + "id"))
            package.content(target, package.xml(target), out, f"slide {index}")
            for rel in package.relations(target).values():
                if rel.get("Type", "").rsplit("/", 1)[-1] == "notesSlide":
                    notes = package.target(target, rel["Id"])
                    package.content(notes, package.xml(notes), out, f"slide {index} notes")
    else:
        source = "xl/workbook.xml"
        root = package.xml(source)
        if root.tag != S + "workbook":
            raise DocumentError("Not a supported XLSX workbook.")
        sheets = root.findall(f"{S}sheets/{S}sheet")
        if not 1 <= len(sheets) <= MAX_PAGES:
            raise DocumentError("Workbook exceeds four sheets or is empty.")
        strings = []
        if "xl/sharedStrings.xml" in package.names:
            strings = ["".join(t.text or "" for child in n if child.tag != S + "rPh" for t in child.iter(S + "t"))
                       for n in package.xml("xl/sharedStrings.xml")]
        out.text("cell values", "Cell values include hidden sheets/cells and are unformatted; dates may be numeric serials. Formulas include their cached value, which may be stale.")
        for sheet in sheets:
            target = package.target(source, sheet.get(R + "id"))
            tree = package.xml(target)
            for cell in tree.iter(S + "c"):
                coordinate = cell.get("r", "?")
                kind = cell.get("t")
                value = cell.findtext(S + "v", "")
                if kind == "s":
                    try:
                        index = int(value)
                        if index < 0: raise ValueError()
                        value = strings[index]
                    except (ValueError, IndexError):
                        raise DocumentError("Invalid shared string reference.") from None
                elif kind == "inlineStr":
                    value = "".join(n.text or "" for n in cell.iter(S + "t"))
                formula = cell.findtext(S + "f")
                if formula is not None:
                    value = f"Formula (not executed): {formula}; cached value: {value or '[absent]'}"
                out.text(f"sheet {sheet.get('name', '?')}; cell {coordinate}", value)
            for drawing in tree.iter(S + "drawing"):
                drawing_path = package.target(target, drawing.get(R + "id"))
                for anchor in package.xml(drawing_path):
                    start = next((n for n in anchor if n.tag.endswith("}from")), None)
                    coordinates = ", ".join(f"{n.tag.rsplit('}', 1)[-1]}={n.text}" for n in start) if start is not None else "absolute position"
                    package.content(drawing_path, anchor, out, f"sheet {sheet.get('name', '?')}; drawing anchor (zero-based) {coordinates}")


def _decode(data, mime):
    from .media import MAX_ASSET_BYTES
    if mime not in DOCUMENT_TYPES or not 0 < len(data) <= MAX_ASSET_BYTES:
        raise DocumentError("Use a PDF, DOCX, PPTX or XLSX file of at most 8 MiB.")
    out = Output(hashlib.sha256(data).hexdigest())
    if mime == PDF:
        _pdf(data, out)
    else:
        _office(data, mime, out)
    return out.parts


def extract_document(data, mime):
    """One subprocess per file; bounded disk output, time and concurrent workers."""
    from .cancellation import check_cancelled
    from .media import ImagePart, MediaError, MAX_ASSET_BYTES
    if mime not in DOCUMENT_TYPES or not 0 < len(data) <= MAX_ASSET_BYTES:
        raise MediaError("Use a PDF, DOCX, PPTX or XLSX file of at most 8 MiB.")
    check_cancelled()
    if not _WORKERS.acquire(blocking=False):
        raise MediaError("Document decoders are busy; try again later.")
    try:
        with tempfile.TemporaryFile() as source, tempfile.TemporaryFile() as output:
            source.write(data); source.seek(0)
            env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "TMPDIR", "PYTHONPATH") if k in os.environ}
            process = subprocess.Popen([sys.executable, "-m", "deepseek_v4_ssd.documents", "--worker", mime],
                stdin=source, stdout=output, stderr=subprocess.DEVNULL, env=env)
            try:
                deadline, next_sample = time.monotonic() + 30, 0
                while process.poll() is None:
                    check_cancelled()
                    now = time.monotonic()
                    if now >= deadline:
                        raise MediaError("Document decoding timed out.")
                    if now >= next_sample:
                        # macOS does not provide a reliable RLIMIT_AS heap cap.
                        # Observe RSS and kill on breach; this is not a hard VM sandbox.
                        probe = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(process.pid)],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=1)
                        if (probe.returncode or not probe.stdout.strip()) and process.poll() is None:
                            raise MediaError("Document memory monitoring failed.")
                        if probe.stdout.strip() and int(probe.stdout) * 1024 > MAX_WORKER_RSS:
                            raise MediaError("Document decoder exceeded its memory budget.")
                        next_sample = now + .25
                    time.sleep(.05)
                check_cancelled()
                if process.returncode != 0:
                    raise MediaError("Document decoder failed or exceeded its resource budget.")
                output.seek(0)
                raw = output.read(MAX_OUTPUT_BYTES + 1)
                if len(raw) > MAX_OUTPUT_BYTES:
                    raise MediaError("Document output exceeds its limit.")
                result = json.loads(raw)
                if "error" in result:
                    raise MediaError(result["error"])
                parts = []
                for part in result["parts"]:
                    check_cancelled()
                    if part["type"] == "text":
                        parts.append(part["text"])
                    else:
                        parts.append(ImagePart.from_bytes(base64.b64decode(part["data"], validate=True), part["mime"]))
                return tuple(parts)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
    finally:
        _WORKERS.release()


def _worker():
    import resource
    from .media import MAX_ASSET_BYTES
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    try:
        parts = _decode(sys.stdin.buffer.read(MAX_ASSET_BYTES + 1), sys.argv[2])
        result = {"parts": parts}
    except DocumentError as error:
        result = {"error": str(error)}
    except Exception:
        result = {"error": "The document is malformed, encrypted, or unsupported."}
    encoded = json.dumps(result, ensure_ascii=False).encode()
    if len(encoded) > MAX_OUTPUT_BYTES:
        encoded = b'{"error":"Document output exceeds its limit."}'
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    if peak > MAX_WORKER_RSS:
        encoded = b'{"error":"Document decoder exceeded its memory budget."}'
    sys.stdout.buffer.write(encoded)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker" or sys.argv[2] not in DOCUMENT_TYPES:
        raise SystemExit("Internal document worker")
    _worker()
