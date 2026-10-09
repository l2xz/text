from pathlib import Path
import hashlib
import shutil
from pypdf import PdfReader
import pypdfium2 as pdfium
from PIL import Image, ImageOps, ImageDraw

source = Path(r'C:\Users\lxz\Desktop\FedCARE (2).pdf')
out = Path(__file__).parent
copy = out / 'FedCARE_current.pdf'
shutil.copy2(source, copy)
reader = PdfReader(copy)
texts = []
for i, page in enumerate(reader.pages, 1):
    text = page.extract_text() or ''
    texts.append(f'\n===== PDF PAGE {i} =====\n{text}\n')
    (out / f'page_{i:02d}.txt').write_text(text, encoding='utf-8')
(out / 'full_text.txt').write_text(''.join(texts), encoding='utf-8')
print('SOURCE:', source)
print('COPY:', copy)
print('SHA256:', hashlib.sha256(source.read_bytes()).hexdigest())
print('PAGES:', len(reader.pages))
print('METADATA:', reader.metadata)
print('FIRST PAGE:', texts[0][:6500])
doc = pdfium.PdfDocument(copy)
thumbs = []
for i in range(len(doc)):
    page = doc[i]
    im = page.render(scale=1.65).to_pil().convert('RGB')
    im.save(out / f'page_{i+1:02d}.png')
    thumb = ImageOps.contain(im, (360, 475))
    tile = Image.new('RGB', (380, 505), 'white')
    tile.paste(thumb, ((380-thumb.width)//2, 25))
    ImageDraw.Draw(tile).text((12, 7), f'PDF page {i+1}', fill='black')
    thumbs.append(tile)
for start in range(0, len(thumbs), 6):
    group = thumbs[start:start+6]
    sheet = Image.new('RGB', (1140, 505*((len(group)+2)//3)), '#dddddd')
    for j, tile in enumerate(group):
        sheet.paste(tile, ((j%3)*380, (j//3)*505))
    sheet.save(out / f'contact_{start+1:02d}.png')
print('Extraction and rendering complete:', out)
