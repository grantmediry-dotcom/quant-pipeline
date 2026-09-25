"""Use packaged DOCX rasterizer with PDFs exported from Microsoft Word.

The packaged LibreOffice converter is unavailable on this Windows runtime.
PDF inputs here are exports of the corresponding current DOCX, not substitutes.
"""
from pathlib import Path
import importlib.util
import os

ROOT = Path(__file__).parent
os.environ['PATH'] = r'C:\Users\grant\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin' + os.pathsep + os.environ['PATH']
path = r'C:\Users\grant\.codex\plugins\cache\openai-primary-runtime\documents\26.909.61513\skills\documents\render_docx.py'
spec = importlib.util.spec_from_file_location('docx_renderer', path)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)

def word_export(doc_path, *args, **kwargs):
    pdf = Path(doc_path).with_suffix('.pdf')
    assert pdf.exists() and pdf.stat().st_mtime >= Path(doc_path).stat().st_mtime
    return str(pdf), 'Current DOCX exported using Microsoft Word'

renderer.convert_to_pdf = word_export
for lang in ['EN', 'CN']:
    doc = ROOT / f'Chihan_Gao_Quant_Resume_{lang}.docx'
    pages = renderer.rasterize(str(doc), str(ROOT / f'qa_{lang}'), 145, False, False)
    print(lang, pages)
