#!/usr/bin/env python3
"""Idempotently migrate authored markup to CSP-compatible event data.

Run after editing/generating model documents and before check_site.py.
No arbitrary handler is executed: security.js accepts a fixed action vocabulary.
"""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[2]
EVENT = re.compile(r'\bon(click|input|change|mouseenter|mouseleave)\s*=')


def migrate(source: str) -> str:
    source = EVENT.sub(lambda match: 'data-call-' + match[1] + '=', source)
    source = re.sub(r"copyTextSafe\(document\.getElementById\('([^']+)'\)\.textContent,\s*'([^']+)'\)",
                    r"copyElementText('\1', '\2')", source)
    source = source.replace("window.scrollTo({top:0,behavior:'smooth'})", 'scrollToTop()')
    return source


def main():
    changed = []
    for path in [ROOT/'biomed-ai-atlas/reference.html', ROOT/'biomed-ai-atlas/reference-runtime.js',
                 ROOT/'biomed-ai-atlas/reference-upgrade.js']:
        if not path.exists():
            continue
        source = path.read_text(encoding='utf-8')
        result = migrate(source)
        if result != source:
            path.write_text(result, encoding='utf-8')
            changed.append(str(path.relative_to(ROOT)))
    for path in sorted((ROOT/'biomed-ai-atlas/models').glob('*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        original = data.get('html')
        if not isinstance(original, str):
            raise ValueError(f'Missing model HTML: {path.name}')
        result = migrate(original)
        if result != original:
            data['html'] = result
            path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':'))+'\n', encoding='utf-8')
            changed.append(str(path.relative_to(ROOT)))
    print(f'Prepared {len(changed)} changed document(s).')


if __name__ == '__main__':
    main()
