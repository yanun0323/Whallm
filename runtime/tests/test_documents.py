import base64
import ctypes
from contextlib import closing
import io
import json
import subprocess
import sys
import threading
import unittest
import zipfile
from unittest.mock import patch

from deepseek_v4_ssd.documents import (PDF, DOCX, PPTX, XLSX, DocumentError, Package,
    MAX_TEXT_BYTES, _decode, extract_document)
from deepseek_v4_ssd.media import (AssetStore, DocumentPart, ImagePart, MediaError,
    ordered_content, expand_documents)
from runtime.tests.test_media import png

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
S = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'


def package(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml', '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        for name, data in files.items(): archive.writestr(name, data)
    return output.getvalue()


def relations(entries):
    return '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + ''.join(
        f'<Relationship Id="{key}" Type="{R}/{kind}" Target="{target}"/>' for key, kind, target in entries) + '</Relationships>'


def docx(text='Invoice total: 42', image=False):
    content = f'<w:p><w:r><w:t>{text}</w:t></w:r></w:p>'
    files = {}
    if image:
        content += '<w:p><w:r><w:drawing><a:blip r:embed="picture"/></w:drawing></w:r></w:p>'
        files['word/_rels/document.xml.rels'] = relations([('picture', 'image', 'media/one.png')])
        files['word/media/one.png'] = png()
    files['word/document.xml'] = f'<w:document xmlns:w="{W}" xmlns:a="{A}" xmlns:r="{R}"><w:body>{content}</w:body></w:document>'
    return package(files)


def pdf(pages=1, text=True, scanned=False, amount=42):
    import pypdfium2 as p
    output = io.BytesIO()
    with p.PdfDocument.new() as document:
        for _ in range(pages):
            with closing(document.new_page(400, 300)) as page:
                if text:
                    obj = p.PdfObject(p.raw.FPDFPageObj_NewTextObj(document, b'Helvetica', 24))
                    buffer = ctypes.create_string_buffer(f'Invoice total: {amount}'.encode('utf-16-le') + b'\0\0')
                    p.raw.FPDFText_SetText(obj, ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ushort)))
                    p.raw.FPDFPageObj_Transform(obj, 1, 0, 0, 1, 40, 150)
                    page.insert_obj(obj)
                    page.gen_content()
                if scanned:
                    from PIL import Image, ImageDraw
                    image = Image.new('RGB', (800, 600), 'white')
                    ImageDraw.Draw(image).text((50, 240), f'Invoice total: {amount}', fill='black', font_size=60)
                    with closing(p.PdfBitmap.from_pil(image)) as bitmap:
                        obj = p.PdfImage.new(document)
                        obj.set_bitmap(bitmap)
                        obj.set_matrix(p.PdfMatrix(a=400, d=300))
                        page.insert_obj(obj)
                        page.gen_content()
        document.save(output)
    return output.getvalue()


class DocumentTests(unittest.TestCase):
    def test_docx_worker_preserves_order_images_and_source_identity(self):
        data = docx(image=True)
        parts = extract_document(data, DOCX)
        text_index = next(i for i, p in enumerate(parts) if isinstance(p, str) and 'Invoice total' in p)
        image_index = next(i for i, p in enumerate(parts) if isinstance(p, ImagePart))
        self.assertLess(text_index, image_index)
        self.assertIn(DocumentPart.from_bytes(data, DOCX).digest, parts[text_index])
        self.assertIn('body block 1', parts[text_index])

    def test_pdf_text_and_no_text_pages_both_have_page_images(self):
        for has_text in (True, False):
            parts = extract_document(pdf(text=has_text, scanned=not has_text), PDF)
            images = [p for p in parts if isinstance(p, ImagePart)]
            self.assertEqual(len(images), 1)
            text = ''.join(p for p in parts if isinstance(p, str))
            self.assertIn('page 1', text)
            self.assertEqual('Invoice total: 42' in text, has_text)
        with self.assertRaisesRegex(MediaError, 'four pages'):
            extract_document(pdf(pages=5), PDF)

    def test_presentation_uses_declared_slide_order_and_notes(self):
        data = package({
            'ppt/presentation.xml': f'<p:presentation xmlns:p="{P}" xmlns:r="{R}"><p:sldIdLst><p:sldId r:id="second"/><p:sldId r:id="first"/></p:sldIdLst></p:presentation>',
            'ppt/_rels/presentation.xml.rels': relations([('first','slide','slides/a.xml'),('second','slide','slides/b.xml')]),
            'ppt/slides/a.xml': f'<p:sld xmlns:p="{P}" xmlns:a="{A}"><a:p><a:t>Alpha</a:t></a:p></p:sld>',
            'ppt/slides/b.xml': f'<p:sld xmlns:p="{P}" xmlns:a="{A}"><a:p><a:t>Beta</a:t></a:p></p:sld>',
            'ppt/slides/_rels/b.xml.rels': relations([('notes','notesSlide','../notesSlides/n.xml')]),
            'ppt/notesSlides/n.xml': f'<p:notes xmlns:p="{P}" xmlns:a="{A}"><a:t>Speaker note</a:t></p:notes>'})
        text = ''.join(p for p in extract_document(data, PPTX) if isinstance(p, str))
        self.assertLess(text.index('Beta'), text.index('Alpha'))
        self.assertIn('slide 1 notes', text)
        self.assertIn('Speaker note', text)

    def test_spreadsheet_coordinates_strings_and_cached_formulas(self):
        data = package({
            'xl/workbook.xml': f'<workbook xmlns="{S}" xmlns:r="{R}"><sheets><sheet name="Orders" r:id="s1"/></sheets></workbook>',
            'xl/_rels/workbook.xml.rels': relations([('s1','worksheet','worksheets/sheet1.xml')]),
            'xl/sharedStrings.xml': f'<sst xmlns="{S}"><si><t>Invoice</t></si></sst>',
            'xl/worksheets/sheet1.xml': f'<worksheet xmlns="{S}"><sheetData><row><c r="A1" t="s"><v>0</v></c><c r="B2"><f>40+2</f><v>42</v></c></row></sheetData></worksheet>'})
        text = ''.join(p for p in extract_document(data, XLSX) if isinstance(p, str))
        self.assertIn('sheet Orders; cell A1', text)
        self.assertIn('Invoice', text)
        self.assertIn('Formula (not executed): 40+2; cached value: 42', text)

    def test_unsafe_archives_xml_and_references_fail_closed(self):
        for name in ('../escape', '/absolute', 'word/vbaProject.bin', 'word/embeddings/ole.bin'):
            with self.subTest(name=name), self.assertRaises(DocumentError): Package(package({name: b'bad'}))
        for encoding in ('utf-8', 'utf-16', 'utf-16-le'):
            xml = '<!DOCTYPE x [<!ENTITY x "expanded">]><x>&x;</x>'.encode(encoding)
            with self.subTest(encoding=encoding), self.assertRaises((DocumentError, UnicodeError)):
                Package(package({'word/document.xml': xml})).xml('word/document.xml')
        files = {'word/document.xml': f'<w:document xmlns:w="{W}"/>',
                 'word/_rels/document.xml.rels': relations([('bad','image','../../private.png')])}
        with self.assertRaises(DocumentError): Package(package(files)).target('word/document.xml', 'bad')
        with self.assertRaises(MediaError): extract_document(b'not a pdf', PDF)
        with self.assertRaisesRegex(MediaError, '64 KiB'): extract_document(docx('x' * MAX_TEXT_BYTES), DOCX)

    def test_cancellation_kills_and_reaps_worker_and_releases_capacity(self):
        from deepseek_v4_ssd.cancellation import cancellation_scope, GenerationCancelled
        from deepseek_v4_ssd import documents
        event = threading.Event()
        children = []
        original = subprocess.Popen
        def spawn(*args, **kwargs):
            child = original(*args, **kwargs)
            children.append(child)
            event.set()
            return child
        with patch.object(documents.subprocess, 'Popen', side_effect=spawn):
            with cancellation_scope(event), self.assertRaises(GenerationCancelled):
                extract_document(docx(), DOCX)
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].returncode)
        self.assertTrue(extract_document(docx(), DOCX))
        documents._WORKERS.acquire(); documents._WORKERS.acquire()
        try:
            with patch.object(documents.subprocess, 'Popen') as spawn:
                with self.assertRaisesRegex(MediaError, 'busy'): extract_document(docx(), DOCX)
                spawn.assert_not_called()
        finally:
            documents._WORKERS.release(); documents._WORKERS.release()

    def test_worker_rss_breach_and_deadline_kill_and_reap(self):
        from deepseek_v4_ssd import documents
        original = subprocess.Popen
        for failure in ('memory budget', 'timed out'):
            children = []
            def spawn(command, **kwargs):
                child = original([sys.executable, '-c', 'import time; time.sleep(10)'], **kwargs)
                children.append(child)
                return child
            with patch.object(documents.subprocess, 'Popen', side_effect=spawn):
                if failure == 'memory budget':
                    with patch.object(documents.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, b'999999999\n')):
                        with self.assertRaisesRegex(MediaError, failure): extract_document(docx(), DOCX)
                else:
                    with patch.object(documents.time, 'monotonic', side_effect=[0, 31]):
                        with self.assertRaisesRegex(MediaError, failure): extract_document(docx(), DOCX)
            self.assertEqual(len(children), 1)
            self.assertIsNotNone(children[0].returncode)

    def test_zip_expansion_and_unsupported_content_are_not_silently_dropped(self):
        with self.assertRaisesRegex(DocumentError, 'expanded bytes'):
            Package(package({'padding': b'x' * (32 * 1024**2 + 1)}))
        for tag in ('chart', 'videoFile', 'oMath', 'AlternateContent'):
            data = package({'word/document.xml': f'<w:document xmlns:w="{W}"><w:body><{tag}/></w:body></w:document>'})
            with self.subTest(tag=tag), self.assertRaises(DocumentError): _decode(data, DOCX)

    def test_asset_adapter_and_request_wide_budgets_precede_decoding(self):
        store = AssetStore()
        try:
            data = docx()
            item = store.put(io.BytesIO(data), len(data), DOCX, 'owner')
            parts = ordered_content([{'type':'text','text':'before'}, {'type':'input_file','file_id':item['id']},
                                     {'type':'text','text':'after'}], asset_store=store, owner='owner')
            self.assertIsInstance(parts[1], DocumentPart)
            result = expand_documents([{'role':'user','content':parts}], enabled=True)
            self.assertTrue(result[0]['content'].startswith('before'))
            self.assertTrue(result[0]['content'].endswith('after'))
            with self.assertRaises(MediaError):
                ordered_content([{'type':'image','file_id':item['id']}], asset_store=store, owner='owner')
            with patch('deepseek_v4_ssd.documents.extract_document') as decode:
                for messages, enabled in (([{'role':'user','content':parts}],False),
                    ([{'role':'assistant','content':parts}],True), ([{'role':'user','content':parts}]*5,True)):
                    with self.assertRaises(MediaError): expand_documents(messages, enabled=enabled)
                decode.assert_not_called()
            store.delete(item['id'], 'owner')
        finally:
            store.close()
