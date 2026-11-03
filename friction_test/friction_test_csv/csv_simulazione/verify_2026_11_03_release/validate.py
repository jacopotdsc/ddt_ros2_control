import json
import sqlite3
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import yaml

root = Path('/home/ubuntu/ddt_ros2_ws/src/friction_test')
out = root / 'friction_test_csv/csv_simulazione/verify_2026_11_03_release'
cfg = yaml.safe_load((root / 'config/friction_test.yaml').read_text())['/**']['ros__parameters']
limits = {j.attrib['name']: j.find('limit').attrib for j in ET.parse(root / 'urdf/robot_friction.xacro').findall('joint') if j.find('limit') is not None}
report = {'files': {}, 'errors': []}
for j in cfg['joints']:
    p = out / (j + '.csv')
    if not p.exists():
        report['errors'].append('missing ' + p.name)
        continue
    a = np.genfromtxt(p, delimiter=',', names=True, dtype=None, encoding='utf-8')
    dt = np.diff(a['t_sim'])
    nans = [k for k in a.dtype.names if a[k].dtype.kind in 'fi' and not np.isfinite(a[k]).all()]
    scalar_matches = all(np.array_equal(a[col], a[prefix + j]) for col, prefix in [('q', 'q_'), ('dq', 'dq_'), ('tau_applied', 'tau_')])
    wheel = j.endswith('4')
    measured_in_limits = True if wheel else bool(np.all((a['q'] >= float(limits[j]['lower'])) & (a['q'] <= float(limits[j]['upper']))))
    levels = sorted(set(int(x) for x in a['speed_level'] if x >= 0))
    segments = sorted(set(a['segment']))
    both_directions = {'speed+', 'speed-'}.issubset(segments) if wheel else {'sweep+', 'sweep-'}.issubset(segments)
    hz = float((len(a) - 1) / (a['t_sim'][-1] - a['t_sim'][0]))
    good = len(a.dtype.names) == 35 and not nans and bool(np.all(dt > 0)) and scalar_matches and measured_in_limits and both_directions and len(levels) == 2 and hz > 450
    report['files'][j] = dict(rows=len(a), duration_s=float(a['t_sim'][-1]-a['t_sim'][0]), hz=hz, max_dt_s=float(dt.max()), nonincreasing=int(sum(dt <= 0)), nonfinite=nans, scalar_matches_full_state=scalar_matches, measured_in_urdf_limits=measured_in_limits, q_min=float(a['q'].min()), q_max=float(a['q'].max()), dq_min=float(a['dq'].min()), dq_max=float(a['dq'].max()), tau_abs_max=float(np.max(np.abs(a['tau_applied']))), levels=levels, segments=segments, passed=bool(good))
    if not good: report['errors'].append('invalid ' + j)
for p in (out / 'command_bag').glob('*.db3'):
    if not (p.parent / 'metadata.yaml').exists():
        raise RuntimeError('Stop the bag recorder before validation')
    con = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
    times = np.array([r[0] for r in con.execute("SELECT m.timestamp FROM messages m JOIN topics t ON m.topic_id=t.id WHERE t.name='/command/joint_cmd' ORDER BY m.timestamp")], dtype=np.int64)
    dt = np.diff(times) * 1e-9
    report['command_recording'] = dict(messages=len(times), duration_s=float((times[-1]-times[0])*1e-9), max_gap_s=float(dt.max()), p99_gap_s=float(np.percentile(dt,99)), gaps_over_timeout=int(sum(dt > 0.2)), hz=float((len(times)-1)/((times[-1]-times[0])*1e-9)))
    if report['command_recording']['hz'] < 200: report['errors'].append('command frequency below 200 Hz')
    if np.any(dt > .2): report['errors'].append('command gap above timeout')
report['passed'] = not report['errors']
(out / 'validation.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
sys.exit(0 if report['passed'] else 1)
