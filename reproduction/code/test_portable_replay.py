"""Portable root relocation and integrity-guard regression checks (tiny fixtures)."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rfinal_replay import guards

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def main():
    obs = module('observation_driver', HERE / 'c17_cache_observations/replay.py')
    sup = module('supplement_driver', HERE / 'c17_replay_supplements_b/replay.py')
    with tempfile.TemporaryDirectory(prefix='portable-replay-') as temp:
        root = Path(temp) / 'renamed_project'
        analysis = root / 'renamed_analysis'
        target = analysis / 'inputs/value.json'
        target.parent.mkdir(parents=True)
        target.write_text('[1, 2, 3]\n')
        project_input = root / 'Imports/value.json'
        project_input.parent.mkdir()
        project_input.write_text('[4]\n')
        contexts = [obs.Context(analysis, root/'out', 'all', 0, 1, None, None),
                    sup.Context(analysis, root/'out', root/'pins.json')]
        for ctx in contexts:
            assert ctx.resolve_recorded('/fixture/old/repro_paper2_20260915/inputs/value.json') == target
            assert ctx.resolve_recorded('/fixture/old/Imports/value.json') == project_input
            assert ctx.resolve_recorded('R/inputs/value.json') == target
            assert ctx.resolve_recorded('P/Imports/value.json') == project_input
            try:
                ctx.resolve_recorded('/fixture/unrelated/missing.json')
            except (obs.ReplayRefusal, sup.ReplayRefusal):
                pass
            else:
                raise AssertionError('unrelated absolute path accepted')
        pins = root / 'pins.json'
        pins.write_text(json.dumps({'files':[{'root':'R','relativepath':'inputs/value.json',
            'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}]}))
        expected = obs.load_expected_inputs(pins, analysis)
        assert obs.verify_expected_inputs(expected)['all_ok']
        target.write_text('tampered\n')
        try:
            obs.verify_expected_inputs(expected)
        except obs.HashRefusal:
            pass
        else:
            raise AssertionError('tampered input accepted')
        target.unlink()
        try:
            obs.verify_expected_inputs(expected)
        except obs.MissingRefusal:
            pass
        else:
            raise AssertionError('missing input accepted')
        populated = root / 'populated'
        populated.mkdir()
        (populated/'existing').write_text('retain')
        try:
            guards.assert_empty_output_dir(populated)
        except guards.ReplayRefusal:
            pass
        else:
            raise AssertionError('nonempty output accepted')
    config = guards.load_config(HERE.parent / 'config/CT_C17_REPLAY_FINAL_v1.relocated.json')
    guards.assert_authorized(config)
    guards.assert_pattern_exceptions_approved(config)
    assert len(config['regeneration_plan']['stages']) == 10
    print('PASS: relocated roots, token paths, unrelated-path refusal, missing/tampered inputs, nonempty output, active config')

if __name__ == '__main__':
    main()
