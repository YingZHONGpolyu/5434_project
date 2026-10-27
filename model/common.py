"""预处理、训练、推理共用的路径、文本清理及运行信息工具。"""

import hashlib
import html
import json
import os
import platform
import re
import socket
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'model' / 'artifacts'
TEXT_VERSION = 'html-nfc-whitespace-v1'
_TAGS = re.compile(r'<\s*/?\s*[a-zA-Z][^>]*>')
_SPACE = re.compile(r'\s+')


def clean_text(value):
    """与 data/clean_data.py 一致；同一函数用于训练集和测试集。"""
    value = html.unescape(_TAGS.sub(' ', value or ''))
    value = unicodedata.normalize('NFC', value)
    value = ''.join(c for c in value if c in '\n\r\t' or unicodedata.category(c) != 'Cc')
    return _SPACE.sub(' ', value).strip()


def write_json(path, value):
    """原子更新报告，避免中断后留下半个 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def manifest_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def hardware_info():
    """记录实际执行机器的信息，不把开发机信息冒充 GPU 服务器。"""
    cpu = platform.processor()
    if Path('/proc/cpuinfo').exists():
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip()
                break
    elif platform.system() == 'Darwin':
        cpu = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    return {
        'hostname': socket.gethostname(), 'cpu_model': cpu,
        'logical_cpu_threads': os.cpu_count(), 'platform': platform.platform(),
        'python': platform.python_version(),
    }
