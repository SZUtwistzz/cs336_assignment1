import csv
import json

import pytest

from cs336_basics.train_LM.plot_training import load_history


def make_run(path, rows, **config):
    path.mkdir()
    (path / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    with (path / 'metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'elapsed_seconds', 'split', 'loss', 'lr'])
        writer.writerows(rows)
    return path


def test_resume_discards_old_future_and_adds_time(tmp_path):
    old = make_run(tmp_path / 'old', [
        [10, 100, 'train', 3, .001], [10, 110, 'valid', 2.8, .001],
        [20, 200, 'train', 2, .001],
    ])
    new = make_run(tmp_path / 'new', [[20, 90, 'train', 1.9, .0001]],
                   resume=str(old / 'checkpoints' / 'step_10.pt'), start_step=10)
    rows = load_history(new)
    assert [r['step'] for r in rows] == [10, 10, 20]
    assert [r['elapsed_seconds'] for r in rows] == [100, 110, 200]
    assert rows[-1]['loss'] == 1.9


def test_missing_checkpoint_time_is_not_silently_invented(tmp_path):
    old = make_run(tmp_path / 'old', [[5, 50, 'train', 3, .001]])
    new = make_run(tmp_path / 'new', [[20, 90, 'train', 2, .001]],
                   resume=str(old / 'checkpoints' / 'step_10.pt'), start_step=10)
    with pytest.raises(ValueError, match='checkpoint step 10'):
        load_history(new)


def test_resume_cycle_is_rejected(tmp_path):
    path = tmp_path / 'cycle'
    make_run(path, [[20, 90, 'train', 2, .001]],
             resume=str(path / 'checkpoints' / 'step_10.pt'), start_step=10)
    with pytest.raises(ValueError, match='cycle'):
        load_history(path)
