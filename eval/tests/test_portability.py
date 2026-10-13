"""Regressions for relocating raw data, prepared corpora and ongoing runs."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from embodied_eval.common import load_corpus, read_json, read_jsonl, write_json, write_jsonl
from embodied_eval.inputs import prepare
from embodied_eval.serving import run, solve_messages
from embodied_eval.scoring import score
import test_framework as framework


RESOURCES = Path(__file__).resolve().parents[1] / 'embodied_eval'


class PortabilityTests(unittest.TestCase):
    def test_move_dataset_corpus_and_partial_run_then_resume(self):
        # Reuse the existing localhost mock; both calls remain deterministic.
        fixture = framework.FrameworkTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        self.addCleanup(fixture.doCleanups)
        original = fixture.root / 'original'
        (original / 'images').mkdir(parents=True)
        shutil.copyfile(fixture.image, original / 'images/frame.png')
        records = [fixture.record(), fixture.record(reference='second')]
        for i, record in enumerate(records, 1):
            record['id'] = f'q{i:06d}'
            record['image_list'][0]['image'] = 'images/frame.png'
        dataset = original / 'benchmark_data.jsonl'
        write_jsonl(dataset, records)
        corpus = original / 'artifacts/corpus'
        prepare(dataset, RESOURCES/'task_catalog.json', RESOURCES/'prompts.json', corpus)
        endpoint, received = fixture.server([(200, fixture.response('A'))])
        config = fixture.root / 'model.json'
        write_json(config, fixture.config(endpoint))
        run_dir = original / 'artifacts/run'
        run(corpus, run_dir, config, limit=1)
        frozen = (corpus / 'items.jsonl').read_bytes()
        contract = read_json(run_dir / 'run.json')['contract']
        moved = fixture.root / 'different/location/release'
        moved.parent.mkdir(parents=True)
        shutil.move(str(original), moved)
        self.assertFalse(original.exists())
        corpus, run_dir = moved / 'artifacts/corpus', moved / 'artifacts/run'
        self.assertEqual((corpus / 'items.jsonl').read_bytes(), frozen)
        run(corpus, run_dir, config)
        self.assertEqual(read_json(run_dir / 'run.json')['contract'], contract)
        self.assertEqual(len(received), 2)
        self.assertEqual(score(corpus, run_dir, moved/'artifacts/scores')['closed_micro']['accuracy'], 0.5)
        self.assertNotIn(str(fixture.root), json.dumps(received))
        fresh = moved / 'artifacts/reprepared'
        prepare(moved/'benchmark_data.jsonl', RESOURCES/'task_catalog.json', RESOURCES/'prompts.json', fresh)
        self.assertEqual(read_jsonl(fresh/'items.jsonl'), read_jsonl(corpus/'items.jsonl'))
        self.assertEqual(read_json(fresh/'manifest.json'), read_json(corpus/'manifest.json'))

    def test_runtime_media_root_never_changes_frozen_hash(self):
        fixture = framework.FrameworkTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        corpus = fixture.corpus()
        _, items = load_corpus(corpus)
        disk = read_jsonl(corpus/'items.jsonl')
        self.assertEqual(items, disk)
        self.assertEqual(json.dumps(items), json.dumps(disk))
        self.assertTrue(solve_messages(items[0])[1]['content'][1]['image_url']['url'].startswith('data:image/png;base64,'))

    def test_path_escape_rejected(self):
        fixture = framework.FrameworkTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        dataset = fixture.root/'dataset/questions.jsonl'
        record = fixture.record()
        record['image_list'][0]['image'] = '../frame.png'
        write_jsonl(dataset, [record])
        with self.assertRaisesRegex(ValueError, 'inside the dataset'):
            prepare(dataset, RESOURCES/'task_catalog.json', RESOURCES/'prompts.json', fixture.root/'corpus')

    def test_stable_ids_survive_task_filtering(self):
        fixture = framework.FrameworkTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        rows = [fixture.record(), fixture.record('step_completion', 'Yes')]
        for i, row in enumerate(rows, 1):
            row['id'] = f'q{i:06d}'
        dataset = fixture.root/'questions.jsonl'
        write_jsonl(dataset, rows)
        corpus = fixture.root/'filtered'
        prepare(dataset, RESOURCES/'task_catalog.json', RESOURCES/'prompts.json', corpus, tasks=['step_completion'])
        self.assertEqual(read_jsonl(corpus/'items.jsonl')[0]['id'], 'q000002')
        self.assertFalse(read_json(corpus/'manifest.json')['full_dataset'])


if __name__ == '__main__':
    unittest.main()
