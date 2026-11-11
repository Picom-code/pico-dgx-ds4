#!/usr/bin/env python3
"""Check a running video-enabled chat endpoint with a local test clip (stdlib + Pillow)."""
import argparse
import base64
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('url', help='chat-completions URL')
parser.add_argument('clip', type=Path)
parser.add_argument('--expected', nargs='+', help='optional codes to read from successive scenes, in order')
parser.add_argument('--max-frames', type=int, default=32)
parser.add_argument('--key-file', type=Path)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
headers = {'Content-Type': 'application/json'}
if args.key_file:
    headers['Authorization'] = 'Bearer ' + args.key_file.read_text().strip()
video = {'type': 'video_url', 'video_url': {'url': 'data:video/mp4;base64,' + base64.b64encode(args.clip.read_bytes()).decode(), 'max_frames': args.max_frames}}
question = 'Read the codes shown in successive scenes in chronological order. Reply with just the codes.' if args.expected else 'Describe what happens in this clip, in chronological order.'
body = {'messages': [{'role': 'user', 'content': [video, {'type': 'text', 'text': question}]}],
        'temperature': 0, 'max_tokens': 256, 'reasoning_effort': 'none'}
results = []


def call(name, request_body, want_status=200):
    request = urllib.request.Request(args.url, json.dumps(request_body).encode(), headers)
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            status = response.status
            if request_body.get('stream'):
                text, done = '', False
                for line in response:
                    if not line.startswith(b'data:'):
                        continue
                    part = line[5:].strip()
                    if part == b'[DONE]':
                        done = True
                        break
                    event = json.loads(part)
                    assert 'error' not in event, event
                    for choice in event.get('choices', []):
                        text += choice.get('delta', {}).get('content') or ''
                assert done, 'stream did not finish with [DONE]'
                data = {'text': text}
            else:
                data = json.load(response)
    except urllib.error.HTTPError as error:
        status, data = error.code, json.load(error)
    assert status == want_status, (name, status, data)
    text = data.get('text', data.get('choices', [{}])[0].get('message', {}).get('content', ''))
    if status == 200:
        assert text.strip(), (name, data)
    row = {'test': name, 'status': status, 'answer': text, 'seconds': round(time.monotonic() - start, 2)}
    if status != 200:
        row['error'] = data.get('error')
    results.append(row)
    print(json.dumps(row), flush=True)
    return text


def check_order(text):
    if args.expected:
        end = 0
        for code in args.expected:
            at = text.find(code, end)
            assert at >= end, (args.expected, text)
            end = at + len(code)


check_order(call('video_non_streaming', body))
s = json.loads(json.dumps(body))
s['stream'] = True
s['messages'][0]['content'][0]['video_url'].pop('max_frames')  # also check the default sampling
check_order(call('video_streaming_default_sampling', s))
for name, url in [('broken_video', 'data:video/mp4;base64,AAAA'), ('private_video_url', 'http://127.0.0.1/private.mp4')]:
    b = json.loads(json.dumps(body))
    b['messages'][0]['content'][0]['video_url']['url'] = url
    call(name, b, 400)
b = json.loads(json.dumps(body))
b['messages'][0]['content'][0]['video_url']['max_frames'] = 101
call('invalid_frame_count', b, 400)
b = json.loads(json.dumps(body))
b['messages'][0]['content'] = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}] * 100 + b['messages'][0]['content']
call('shared_100_frame_budget', b, 400)
image = io.BytesIO()
Image.new('RGB', (512, 512), 'red').save(image, format='PNG')
image_url = 'data:image/png;base64,' + base64.b64encode(image.getvalue()).decode()
b = {'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': image_url}}, {'type': 'text', 'text': 'What single color fills this image? Reply with only the color name.'}]}], 'reasoning_effort': 'none', 'max_tokens': 32, 'temperature': 0}
assert 'red' in call('image_after_video', b).lower()
b = {'messages': [{'role': 'user', 'content': 'What is 6 multiplied by 7? Reply with only the integer.'}], 'reasoning_effort': 'none', 'max_tokens': 32, 'temperature': 0}
assert call('text_after_video', b).strip() == '42'
if args.output:
    args.output.write_text(json.dumps(results, indent=2) + '\n')
print('PASS: video streaming, scene order, errors, shared frame budget, image and text requests', flush=True)
