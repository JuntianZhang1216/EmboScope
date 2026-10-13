"""The code archive has a fixed empty data slot; the download fills that slot."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image
from embodied_eval import layout
from embodied_eval.common import file_hash, write_json, write_jsonl
from embodied_eval.inputs import inspect_image


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('data_layout_packer', ROOT / 'eval/scripts/pack_release.py')
packer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packer)


class DataLayoutTests(unittest.TestCase):
    def test_installed_package_finds_checkout_from_child_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'checkout'
            (root / 'eval/embodied_eval').mkdir(parents=True)
            (root / 'evaluate.sh').touch()
            (root / 'pyproject.toml').touch()
            child = root / 'outputs/run'
            child.mkdir(parents=True)
            with patch.object(layout, '__file__', str(Path(folder) / 'lib/site-packages/embodied_eval/layout.py')):
                with patch.object(Path, 'cwd', return_value=child):
                    self.assertEqual(layout.project_root(), root)
                with patch.object(Path, 'cwd', return_value=Path(folder)):
                    with self.assertRaisesRegex(ValueError, 'EmboScope checkout'):
                        layout.project_root()

    def test_code_and_data_archives_round_trip_into_fixed_layout(self):
        with tempfile.TemporaryDirectory(prefix='fixed data layout ') as folder:
            base = Path(folder)
            publisher = base / 'publisher'
            data = publisher / 'data'
            (data / 'images').mkdir(parents=True)
            image = data / 'images/frame.png'
            Image.new('RGB', (8, 8), (10, 20, 30)).save(image)
            inspected = inspect_image(image)
            record = dict(id='q000001', task='relative_distance', text_list=['A. first\nB. second', 'first'],
                          role_list=['human', 'human', 'gpt'], image_list=[dict(image='images/frame.png',
                          matched_text_index=0, **{k: inspected[k] for k in ('sha256', 'width', 'height')})])
            write_jsonl(data / 'benchmark_data.jsonl', [record])
            # Non-dataset files placed in data/ must not enter the download.
            (data / '.gitkeep').touch()
            (data / 'local-note.txt').write_text('not for release')
            code_zip, data_tar = base / 'code.zip', base / 'data.tar.gz'
            packer.pack_code(ROOT, code_zip)
            self.assertEqual(packer.pack_data(publisher, data_tar), 1)
            checkout = base / 'downloaded checkout'
            with zipfile.ZipFile(code_zip) as archive:
                self.assertEqual([n for n in archive.namelist() if n.startswith('data/')], ['data/.gitkeep'])
                archive.extractall(checkout)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(checkout / 'eval'))
            command = ['bash', str(checkout / 'evaluate.sh'), '--python', sys.executable,
                       '--stage', 'prepare', '--output', str(base / 'corpus run')]
            absent = subprocess.run(command, cwd=base, env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(absent.returncode, 2)
            self.assertIn('missing data/benchmark_data.jsonl', absent.stderr)
            with tarfile.open(data_tar) as archive:
                self.assertEqual(set(archive.getnames()), {'benchmark_data.jsonl', 'images/frame.png'})
                for member in archive.getmembers():
                    self.assertTrue(member.isfile())
                    self.assertFalse(Path(member.name).is_absolute())
                    self.assertNotIn('..', Path(member.name).parts)
                    self.assertEqual(member.uid, 0)
                    self.assertEqual(member.uname, '')
                archive.extractall(checkout / 'data')
            prepared = subprocess.run(command, cwd=base, env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(prepared.returncode, 0, prepared.stdout + prepared.stderr)
            self.assertFalse((checkout / 'data/metadata').exists())
            # Full validation reads the code checkout's metadata, not downloaded metadata.
            metadata = checkout / 'metadata'
            write_jsonl(metadata / 'images.jsonl', [dict(inspected, path='images/frame.png')])
            write_jsonl(metadata / 'video_frames.jsonl', [])
            manifest = dict(dataset='data/benchmark_data.jsonl', dataset_sha256=file_hash(data / 'benchmark_data.jsonl'),
                            images_manifest_sha256=file_hash(metadata / 'images.jsonl'),
                            video_frames_sha256=file_hash(metadata / 'video_frames.jsonl'), questions=1,
                            tasks={'relative_distance': 1}, answer_formats={'single_choice': 1},
                            unique_images=1, media_references=1, video_questions_as_fixed_frames=0,
                            media_bytes=inspected['bytes'])
            write_json(metadata / 'release_manifest.json', manifest)
            validated = subprocess.run([sys.executable, '-m', 'embodied_eval', 'validate'], cwd=base,
                                       env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(validated.returncode, 0, validated.stdout + validated.stderr)
            self.assertEqual(json.loads(validated.stdout)['status'], 'passed')


if __name__ == '__main__':
    unittest.main()
