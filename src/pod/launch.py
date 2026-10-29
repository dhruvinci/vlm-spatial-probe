"""Rent one GPU on RunPod, run src/pod/run_all.sh on it, fetch each model's cache as it finishes, then terminate.

The API key is read from ~/.runpod/config.toml (key `apikey`) and never printed. Spend is tracked two ways, the
account balance drop and the pod's hourly price times its uptime; the watchdog terminates the pod when either reaches
--stop-at (default $9.00, under the $10 cap, leaving room for the last download).

  python src/pod/launch.py create   --state S.json --pubkey KEY.pub
  python src/pod/launch.py upload   --state S.json --key KEY --bundle BUNDLE.tar
  python src/pod/launch.py start    --state S.json --key KEY
  python src/pod/launch.py watch    --state S.json --key KEY --cache DIR [--stop-at 9.0]   (fetches, then terminates)
  python src/pod/launch.py ssh      --state S.json --key KEY -- CMD
  python src/pod/launch.py terminate --state S.json
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

IMAGE = 'runpod/pytorch:1.4.0-cu1290-torch291-ubuntu2404'
GPUS = ['NVIDIA A100 80GB PCIe', 'NVIDIA A100-SXM4-80GB', 'NVIDIA RTX PRO 6000 Blackwell Server Edition',
        'NVIDIA RTX PRO 6000 Blackwell Workstation Edition']
REMOTE = '/workspace/p04'


def key():
    txt = pathlib.Path.home().joinpath('.runpod/config.toml').read_text()
    return re.search(r'apikey\s*=\s*(.+)', txt).group(1).strip().strip('\'"').strip()


def call(url, method='GET', payload=None, gql=False):
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(), method=method,
                                 headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key()}',
                                          'User-Agent': 'runpod-python/1.7.0'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            b = r.read()
            return json.loads(b) if b else {}
    except urllib.error.HTTPError as e:
        return {'error': e.code, 'body': e.read().decode()[:600]}


def rest(path, method='GET', payload=None):
    return call('https://rest.runpod.io/v1' + path, method, payload)


def account():
    q = 'query { myself { clientBalance currentSpendPerHr pods { id desiredStatus costPerHr } } }'
    return call('https://api.runpod.io/graphql', 'POST', {'query': q})['data']['myself']


def load(p):
    return json.loads(pathlib.Path(p).read_text())


def save(p, d):
    pathlib.Path(p).write_text(json.dumps(d, indent=1))


def ssh_base(st, k):
    return ['ssh', '-i', k, '-p', str(st['port']), '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=30', '-o', 'LogLevel=ERROR', f"root@{st['ip']}"]


def ssh(st, k, cmd, **kw):
    return subprocess.run(ssh_base(st, k) + [cmd], capture_output=True, text=True, timeout=kw.get('timeout', 120))


def spend(st):
    acc = account()
    drop = st['balance0'] - float(acc['clientBalance'])
    est = (time.time() - st['created']) / 3600 * st.get('cost_per_hr', 0)
    return drop, est, acc


def cmd_create(a):
    acc = account()
    if acc['pods']:
        sys.exit(f"refusing: account already has pods {[p['id'] for p in acc['pods']]}")
    payload = {'name': 'vlm-spatial-probe', 'cloudType': 'COMMUNITY', 'computeType': 'GPU', 'gpuCount': 1,
               'gpuTypeIds': GPUS, 'gpuTypePriority': 'custom', 'allowedCudaVersions': ['12.9', '13.0'],
               'imageName': IMAGE, 'containerDiskInGb': 200, 'volumeInGb': 0, 'ports': ['22/tcp'], 'supportPublicIp': True,
               'interruptible': False, 'minRAMPerGPU': 32, 'minVCPUPerGPU': 6,
               'env': {'PUBLIC_KEY': pathlib.Path(a.pubkey).read_text().strip()}}
    t0 = time.time()
    pod = rest('/pods', 'POST', payload)
    if 'id' not in pod:
        sys.exit(f'create failed: {pod}')
    st = {'pod_id': pod['id'], 'created': t0, 'balance0': float(acc['clientBalance']), 'cost_per_hr': float(pod.get('costPerHr') or 0),
          'gpu': (pod.get('machine') or {}).get('gpuTypeId') or pod.get('gpuTypeId'), 'image': IMAGE}
    save(a.state, st)
    print('created', st['pod_id'], 'cost/hr', st['cost_per_hr'], 'gpu', st['gpu'], flush=True)
    for _ in range(90):                                         # wait for a public IP and the SSH port
        p = rest(f"/pods/{st['pod_id']}")
        pm = p.get('portMappings') or {}
        if p.get('publicIp') and pm.get('22'):
            st.update(ip=p['publicIp'], port=int(pm['22']), cost_per_hr=float(p.get('costPerHr') or st['cost_per_hr']),
                      gpu=(p.get('machine') or {}).get('gpuDisplayName') or st['gpu'])
            save(a.state, st)
            break
        time.sleep(10)
    else:
        print('no public IP / SSH port after 15 min; terminating', flush=True)
        cmd_terminate(a)
        sys.exit(1)
    print('ssh ready at', st['ip'], st['port'], 'gpu', st['gpu'], 'cost/hr', st['cost_per_hr'], flush=True)


def cmd_upload(a):
    st = load(a.state)
    for _ in range(30):
        if ssh(st, a.key, 'mkdir -p ' + REMOTE).returncode == 0:
            break
        time.sleep(10)
    r = subprocess.run(['scp', '-i', a.key, '-P', str(st['port']), '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null',
                        '-o', 'LogLevel=ERROR', a.bundle, f"root@{st['ip']}:{REMOTE}/bundle.tar"], capture_output=True, text=True)
    print('scp', r.returncode, r.stderr[-300:])
    print(ssh(st, a.key, f'cd {REMOTE} && tar xf bundle.tar && rm bundle.tar && ls').stdout)


def cmd_start(a):
    st = load(a.state)
    r = ssh(st, a.key, f'cd {REMOTE} && nohup bash src/pod/run_all.sh > /dev/null 2>&1 & echo started')
    print(r.stdout, r.stderr[-300:])


def fetch(st, k, name, cache, streams=4):
    """Copy out/<name> from the pod, skipping files already here with the same size; up to `streams` files at once."""
    dest = pathlib.Path(cache)
    (dest/name).mkdir(parents=True, exist_ok=True)
    t = time.time()
    r = ssh(st, k, f"cd {REMOTE}/out && find {name} -type f -printf '%s %p\\n'")
    files = [ln.split(' ', 1) for ln in r.stdout.splitlines() if ' ' in ln]
    todo = [f for sz, f in files if not ((dest/f).exists() and (dest/f).stat().st_size == int(sz))]
    procs = []
    for f in sorted(todo, key=lambda f: -int(dict((b, a) for a, b in files)[f])):
        while len([q for q in procs if q.poll() is None]) >= streams:
            time.sleep(2)
        procs.append(subprocess.Popen(' '.join(ssh_base(st, k)) + f" 'tar cf - -C {REMOTE}/out {f}' | tar xf - -C {dest}", shell=True))
    for q in procs:
        q.wait()
    ok = all((dest/f).exists() and (dest/f).stat().st_size == int(sz) for sz, f in files) and bool(files)
    print('fetched' if ok else 'FETCH FAILED', name, len(todo), 'files', round(time.time() - t), 's', flush=True)
    return ok


def fetch_data(st, k, cache):
    """Logs, the layout audits, and the dataset files built on the pod (whichever exist)."""
    paths = ('logs out/E00_audit_*.json data/dataset.json data/nyu.json data/nyu_dense.npz data/images/indoor '
             'data_e03/dataset.json data_e03/hypersim_checks.json data_e03/images')
    p = subprocess.run(' '.join(ssh_base(st, k)) + f" 'cd {REMOTE} && tar cf - $(ls -d {paths} 2>/dev/null)' | tar xf - -C {cache}",
                       shell=True, capture_output=True, text=True, timeout=1800)
    print('fetched logs and data', p.returncode, p.stderr[-300:], flush=True)


def cmd_watch(a):
    st = load(a.state)
    got = set(st.get('fetched', []))
    while True:
        drop, est, acc = spend(st)
        r = ssh(st, a.key, f"cd {REMOTE} && ls out logs 2>/dev/null | tr '\\n' ' '; echo; tail -2 logs/run_all.log; "
                           f"for f in logs/extract_*.log logs/data.log; do [ -f $f ] && echo \"$f: $(tail -c 160 $f | tr '\\n\\r' '  ')\"; done; "
                           f"ls ALL_DONE data/READY 2>/dev/null; df -h /workspace | tail -1")
        out = r.stdout
        print(time.strftime('%H:%M:%S'), f'spend balance-drop ${drop:.2f} uptime-estimate ${est:.2f}', '|', out.replace('\n', ' | ')[-900:], flush=True)
        for name in re.findall(r'(\S+)\.DONE', out):
            if name not in got and fetch(st, a.key, name, a.cache):
                got.add(name); st['fetched'] = sorted(got); save(a.state, st)
        if 'data/READY' in out and not st.get('data_fetched'):
            fetch_data(st, a.key, a.cache); st['data_fetched'] = True; save(a.state, st)
        done = 'ALL_DONE' in out
        if done or max(drop, est) >= a.stop_at:
            if not done:
                print(f'STOP: spend reached ${max(drop, est):.2f} >= ${a.stop_at}; fetching what is done, then terminating', flush=True)
            fetch_data(st, a.key, a.cache)
            cmd_terminate(a)
            return
        time.sleep(a.every)


def cmd_ssh(a):
    st = load(a.state)
    r = ssh(st, a.key, ' '.join(a.rest), timeout=600)
    print(r.stdout[-6000:], r.stderr[-1000:])


def cmd_terminate(a):
    st = load(a.state)
    r = rest(f"/pods/{st['pod_id']}", 'DELETE')
    print('terminate', st['pod_id'], r, flush=True)
    for _ in range(30):
        acc = account()
        if not acc['pods']:
            drop = st['balance0'] - float(acc['clientBalance'])
            st.update(terminated=time.time(), balance_end=float(acc['clientBalance']), spend_balance_drop=round(drop, 4),
                      uptime_h=round((time.time() - st['created']) / 3600, 3))
            save(a.state, st)
            print(f"verified: no pods on the account; balance {acc['clientBalance']:.4f}; spend so far ${drop:.2f}; uptime {st['uptime_h']} h", flush=True)
            return
        time.sleep(10)
    print('WARNING: pod still listed after terminate', acc['pods'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['create', 'upload', 'start', 'watch', 'ssh', 'terminate'])
    ap.add_argument('--state', required=True)
    ap.add_argument('--key')
    ap.add_argument('--pubkey')
    ap.add_argument('--bundle')
    ap.add_argument('--cache')
    ap.add_argument('--stop-at', type=float, default=9.0)
    ap.add_argument('--every', type=int, default=60)
    ap.add_argument('rest', nargs='*')
    a = ap.parse_args()
    {'create': cmd_create, 'upload': cmd_upload, 'start': cmd_start, 'watch': cmd_watch, 'ssh': cmd_ssh,
     'terminate': cmd_terminate}[a.cmd](a)


if __name__ == '__main__':
    main()
