"""Paths and source provenance shared by the follow-up command-line tools."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_DEFAULT = ROOT/'results/continuous_boundary_followup_20260929'
DERIVATIVE_DEFAULT = ROOT/'results/derivative_curves_20260929'


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs(folder, root=ROOT, allow_snapshots=True):
    """Archived sources verify historical provenance, not the current algorithm.

    Training may resume only with current sources. Read-only validation can use
    exact recorded Python sources; data/configuration changes always fail.
    """
    folder, root = Path(folder), Path(root).resolve()
    manifest = json.loads((folder/'input_sha256.json').read_text(encoding='utf-8'))
    archived = []
    for name, expected in manifest.items():
        relative = Path(name.replace('\\', '/'))
        current = (root/relative).resolve()
        if not current.is_relative_to(root):
            raise ValueError(f'Input path escapes project: {name}')
        if current.is_file() and sha256(current) == expected:
            continue
        snapshot = (folder/'source_snapshot'/relative).resolve()
        if (allow_snapshots and current.suffix == '.py'
                and snapshot.is_relative_to((folder/'source_snapshot').resolve())
                and snapshot.is_file() and sha256(snapshot) == expected):
            archived.append(name)
            continue
        raise ValueError(f'Recorded input changed or is missing: {name}')
    return dict(checked=len(manifest), archived_sources=archived)


def snapshot_sources(folder, files, root=ROOT):
    for path in files:
        path = Path(path)
        if path.suffix != '.py':
            continue
        target = Path(folder)/'source_snapshot'/path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
