"""Ejecuta un notebook y exporta HTML con código plegable."""
import os
import sys
import tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ['PATH']=str(ROOT/'.venv/bin')+os.pathsep+os.environ['PATH']
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'cabify-mpl'))
os.environ.setdefault('JUPYTER_RUNTIME_DIR',str(Path(tempfile.gettempdir())/'cabify-jupyter-runtime'))
path=ROOT/(sys.argv[1] if len(sys.argv)>1 else '02_modelado_rutas.ipynb')
nb=nbformat.read(path,as_version=4)
NotebookClient(nb,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
nbformat.validate(nb);nbformat.write(nb,path)
exporter=HTMLExporter(template_name='lab')
exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
body,_=exporter.from_notebook_node(nb)
soup=BeautifulSoup(body,'html.parser')
style=soup.new_tag('style')
style.string="""
body,.jp-Notebook { background:#FAF9F5!important; color:#193441; }
main { max-width:1120px; margin:0 auto; padding:24px 28px 80px!important; }
.jp-MarkdownCell .jp-RenderedHTMLCommon { font:16px/1.72 Arial,sans-serif; color:#193441; }
.jp-RenderedHTMLCommon h1 { font-size:42px; line-height:1.15; letter-spacing:-1px; color:#193441; }
.jp-RenderedHTMLCommon h2 { font-size:27px; line-height:1.3; border-top:1px solid #d4dfe0; padding-top:28px; margin-top:44px; }
.jp-RenderedHTMLCommon h3,.jp-RenderedHTMLCommon a { color:#117c83; }
.jp-RenderedHTMLCommon blockquote { border-left:4px solid #e87524; background:#f1eee5; padding:14px 18px; margin:22px 0; color:#193441; }
.jp-RenderedHTMLCommon table { font-size:13px; line-height:1.5; }
.jp-RenderedHTMLCommon th { background:#193441; color:white; padding:9px; text-align:left; }
.jp-RenderedHTMLCommon td { padding:9px; border-bottom:1px solid #dfe6e7; }
.jp-OutputArea-output img { max-width:100%;height:auto; }
details.code-panel { border:1px solid #dfe6e7; border-radius:6px; margin:8px 0 16px; background:#f2f4f2; }
details.code-panel summary { padding:10px 14px; cursor:pointer; color:#637781; font:13px Arial,sans-serif; }
.jp-InputArea-editor { border:none!important; padding:8px; background:#f2f4f2!important; }
@media(max-width:700px) { main { padding:12px 8px!important; } .jp-RenderedHTMLCommon h1 { font-size:32px; } }
"""
soup.head.append(style)
for cell in soup.select('.jp-CodeCell'):
    wrapper=cell.select_one('.jp-Cell-inputWrapper')
    if wrapper:
        details=soup.new_tag('details',attrs={'class':'code-panel'})
        summary=soup.new_tag('summary');summary.string='Ver código reproducible'
        details.append(summary);wrapper.replace_with(details);details.append(wrapper)
if soup.title:soup.title.string='¿Han seguido la misma ruta? · Modelado y comparación'
path.with_suffix('.html').write_text(str(soup),encoding='utf-8')
print(f'Ejecutado y exportado: {path.name}; {len(nb.cells)} celdas, {len(soup.select("img"))} imágenes.')
