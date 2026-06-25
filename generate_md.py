import json
from pathlib import Path

summary = json.loads(Path('real_tess_validation.json').read_text(encoding='utf-8'))
results = summary['results']

md = ['# Real TESS Validation Results\n']
md.append('| TIC ID | Observations | Period (days) | Depth (%) | Radius (R⊕) | Classification | Confidence |')
md.append('|--------|--------------|---------------|-----------|-------------|----------------|------------|')

for r in results:
    md.append(
        f"| {r['tic_id']} | {r['observations']} | {r['period_days']} | "
        f"{r['transit_depth_percent']} | {r['radius_earth']} | "
        f"{r['classification']} | {r['confidence'] * 100:.1f}% |"
    )
    
Path('docs/real_tess_results.md').parent.mkdir(exist_ok=True)
Path('docs/real_tess_results.md').write_text('\n'.join(md), encoding='utf-8')
print('Markdown generated.')
