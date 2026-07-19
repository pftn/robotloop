import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from robotloop.schema.episode import DataSource, Episode, Step


@pytest.fixture
def make_episode():
    def _make(idx=0, n=10, task="pick_red_cube", embodiment="franka",
              success=True, source=DataSource.SIM, dim=7, fps=10.0):
        steps = [
            Step(
                frame_index=i,
                timestamp=i / fps,
                observation={"images": {}, "state": [0.1 * i] * dim},
                action=[0.01 * i] * dim,
                is_terminal=(i == n - 1),
                language_instruction=task,
            )
            for i in range(n)
        ]
        return Episode(
            task=task,
            language_instruction=task,
            embodiment_tag=embodiment,
            source=source,
            success=success,
            episode_id=f"ep{idx:04d}",
            duration=n / fps,
            dataset_name="test_ds",
            episode_index=idx,
            fps=fps,
            robot_type="panda",
            steps=steps,
        ).validate()

    return _make
