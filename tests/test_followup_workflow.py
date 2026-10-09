"""Regression checks for provenance and complete simulation reporting."""
from pathlib import Path
import csv
import hashlib
import importlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from followup_io import verify_inputs
import followup_continuous_boundary as boundary
import followup_simulation_queue as queue


class ProvenanceTests(unittest.TestCase):
    def test_historical_source_is_allowed_only_for_verification(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'result';out.mkdir()
            (root/'code.py').write_bytes(b'changed')
            snapshot=out/'source_snapshot/code.py';snapshot.parent.mkdir();snapshot.write_bytes(b'original')
            (out/'input_sha256.json').write_text(json.dumps({'code.py':hashlib.sha256(b'original').hexdigest()}))
            self.assertEqual(verify_inputs(out,root)['archived_sources'],['code.py'])
            with self.assertRaises(ValueError):verify_inputs(out,root,allow_snapshots=False)
            snapshot.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_inputs(out,root)

    def test_data_changes_cannot_be_hidden_by_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'result';out.mkdir()
            (root/'data.npy').write_bytes(b'changed')
            snapshot=out/'source_snapshot/data.npy';snapshot.parent.mkdir();snapshot.write_bytes(b'original')
            (out/'input_sha256.json').write_text(json.dumps({'data.npy':hashlib.sha256(b'original').hexdigest()}))
            with self.assertRaises(ValueError):verify_inputs(out,root)

    def test_windows_relative_paths_and_escape_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'result';out.mkdir();(root/'scripts').mkdir()
            (root/'scripts/code.py').write_bytes(b'code')
            path=out/'input_sha256.json'
            path.write_text(json.dumps({'scripts\\code.py':hashlib.sha256(b'code').hexdigest()}))
            self.assertEqual(verify_inputs(out,root)['checked'],1)
            path.write_text(json.dumps({'../escape.py':'unused'}))
            with self.assertRaises(ValueError):verify_inputs(out,root)


class QueueTests(unittest.TestCase):
    def test_import_does_not_start_work(self):
        with patch.object(boundary,'initialize',side_effect=AssertionError('unexpected work')):
            importlib.reload(queue)
        importlib.reload(queue)

    def test_timeout_leaves_no_completed_result(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            with patch.object(queue,'initialize',return_value={'seeds':[1],'modes':['hard','split']}):
                with self.assertRaises(TimeoutError):queue.run(out,timeout=0)
            self.assertFalse((out/'simulation_completion.json').exists())

    def test_global_completion_requires_all_seeds(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);cfg={'seeds':[1,2],'modes':['hard','split']}
            for seed in cfg['seeds']:
                for phase in ['frozen','iteration']:
                    for mode in cfg['modes']:
                        dest=out/phase/f'seed{seed}'/mode;dest.mkdir(parents=True)
                        (dest/'completion.json').write_text('{"status":"complete"}')
            def fake_simulation(folder,settings,report_dir):
                self.assertFalse((folder/'simulation_completion.json').exists())
                report_dir.mkdir(parents=True)
                boundary.write_csv(report_dir/'simulation_summary.csv',[{'seed':settings['seeds'][0],'gain_bp':1}])
            with patch.object(queue,'initialize',return_value=cfg),patch.object(queue,'simulation',side_effect=fake_simulation):
                queue.run(out,timeout=0)
            record=json.loads((out/'simulation_completion.json').read_text())
            self.assertEqual(record['seeds'],[1,2]);self.assertEqual(record['rows'],2)


if __name__=='__main__':unittest.main()
