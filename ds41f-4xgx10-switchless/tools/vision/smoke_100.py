#!/usr/bin/env python3
"""Exercise the live endpoint's vision path and its 100-image boundary."""
import argparse
import base64
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('url', help='chat-completions URL of a running vision-enabled server')
parser.add_argument('--key-file', type=Path, help='optional bearer API key file')
parser.add_argument('--font', default='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
parser.add_argument('--output', type=Path, help='optional JSON results file')
args = parser.parse_args()
URL, FONT = args.url, args.font
HEADERS = {'Content-Type': 'application/json'}
if args.key_file:
    HEADERS['Authorization'] = 'Bearer ' + args.key_file.read_text().strip()
RESULTS = []

def card(index, code):
    image = Image.new('RGB', (768, 768), '#f4f4f4')
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, 755, 755), outline='black', width=8)
    draw.text((384, 235), f'IMAGE {index}', fill='black', font=ImageFont.truetype(FONT, 72), anchor='mm')
    draw.text((384, 435), str(code), fill='#173c91', font=ImageFont.truetype(FONT, 145), anchor='mm')
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()}}

def chat(name, content, expected=None, status=200, **extra):
    body = {'messages': [{'role': 'user', 'content': content}], 'temperature': 0, 'max_tokens': 96,
            'reasoning_effort': 'none', **extra}
    request = urllib.request.Request(URL, json.dumps(body).encode(), HEADERS)
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            actual, data = response.status, json.load(response)
    except urllib.error.HTTPError as error:
        actual, data = error.code, json.load(error)
    assert actual == status, (name, actual, data)
    if status == 200:
        answer = data['choices'][0]['message'].get('content') or ''
        assert expected is None or all(part in answer for part in expected), (name, answer)
        row = {'test': name, 'status': actual, 'answer': answer, 'usage': data.get('usage'),
               'seconds': round(time.monotonic() - start, 2)}
    else:
        row = {'test': name, 'status': actual, 'error': data.get('error'), 'seconds': round(time.monotonic() - start, 2)}
    RESULTS.append(row)
    print(json.dumps(row), flush=True)
    return row

first, last = card(1, 7319), card(100, 8624)
chat('one_image_ocr', [first, {'type': 'text', 'text': 'Read the large four-digit number in the image. Reply with only that number.'}], ['7319'])
chat('two_images_ocr', [first, last, {'type': 'text', 'text': 'Read the large four-digit number in each image. Reply with the first number, then the second number.'}], ['7319', '8624'])
images = [first] + [card(i, 5000+i) for i in range(2, 100)] + [last]
chat('101_images_rejected', images + [first], status=400)
chat('broken_image_rejected', [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}], status=400)
large = chat('100_images_ocr', images + [{'type': 'text', 'text': 'Read the large four-digit code in the FIRST image and the LAST image only. Reply as first code, last code.'}], ['7319', '8624'])
assert large['usage']['prompt_tokens'] > 20000, large
chat('text_after_images', 'What is 17 multiplied by 23? Reply with only the integer.', ['391'])
if args.output:
    args.output.write_text(json.dumps(RESULTS, indent=2) + '\n')
print('PASS: single image, two images, 100 images, 101-image rejection, invalid image, and text afterwards', flush=True)
