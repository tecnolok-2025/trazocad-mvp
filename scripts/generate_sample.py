from pathlib import Path
from PIL import Image, ImageDraw

out = Path(__file__).resolve().parents[1] / 'examples' / 'sample_croquis.png'
out.parent.mkdir(parents=True, exist_ok=True)
img = Image.new('RGB', (1400, 1000), 'white')
d = ImageDraw.Draw(img)

wall = 8
for offset in range(wall):
    d.rectangle((100+offset, 100+offset, 1200-offset, 800-offset), outline='black')

for offset in range(4):
    d.line((650, 100, 650, 800), fill='black', width=3)
    d.line((100, 450, 650, 450), fill='black', width=3)
    d.line((650, 300, 1200, 300), fill='black', width=3)

for x in range(610, 691):
    img.putpixel((x, 450), (255, 255, 255))
for y in range(260, 341):
    img.putpixel((650, y), (255, 255, 255))

for offset in range(3):
    d.rectangle((180+offset, 180+offset, 420-offset, 340-offset), outline='black')
    d.rectangle((760+offset, 130+offset, 1080-offset, 250-offset), outline='black')
    d.rectangle((760+offset, 380+offset, 1080-offset, 650-offset), outline='black')

d.text((150, 60), 'TrazoCad - Croquis ejemplo', fill='black')
d.text((170, 360), 'OFICINA', fill='black')
d.text((800, 110), 'SALA', fill='black')
d.text((800, 350), 'DEPOSITO', fill='black')
d.text((430, 870), 'Referencia de escala: 5000 mm', fill='black')
img.save(out)
print(out)
