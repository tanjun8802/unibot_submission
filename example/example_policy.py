"""HangCup joint-space behavior cloning from LeRobot demonstrations.

Train inside the policy container with PyArrow installed:
    python example/example_policy.py DATASET CHECKPOINT.npz
Set UNIBOT_HANGCUP_CHECKPOINT to the resulting file before starting the server.
PyArrow is only needed for training.
"""

import argparse
import os
from pathlib import Path

import numpy as np

OFFSETS = (-8, -4, 0)
CHUNK_SIZE = 4
CHANNELS = {"left_arm": 7, "right_arm": 7, "left_gripper": 1, "right_gripper": 1}
STATE_KEYS = tuple(f"observation.state.{name}" for name in CHANNELS)
ACTION_KEYS = tuple(f"action.{name}" for name in CHANNELS) + ("action.pivot",)
STATE_DIM = sum(CHANNELS.values())


def features(history, mean, scale, projection, bias):
    normalized = np.clip((history - mean) / scale, -8, 8)
    return np.concatenate(
        (normalized, np.cos(normalized @ projection + bias),
         np.ones((len(history), 1), dtype=np.float32)),
        axis=1,
    )


class ExamplePolicy:
    """Predict a short chunk of joint targets from measured motion history."""

    def __init__(self):
        self._token = os.environ["UNIBOT_SUBMISSION_TOKEN"]
        if os.environ.get("UNIBOT_CONTROL_SPACE") != "joint":
            raise ValueError("HangCup requires UNIBOT_CONTROL_SPACE=joint")
        with np.load(os.environ["UNIBOT_HANGCUP_CHECKPOINT"], allow_pickle=False) as model:
            self._model = {name: model[name] for name in model.files}

    @property
    def metadata(self):
        return {
            "control_space": "joint",
            "obs_delta_indices": {key: list(OFFSETS) for key in STATE_KEYS},
            "image_resize": {},
            "action_chunk_size": CHUNK_SIZE,
            "token": self._token,
        }

    def get_action(self, obs):
        history = np.concatenate([obs[key] for key in STATE_KEYS], axis=-1)
        history = np.asarray(history, dtype=np.float32).reshape(1, -1)
        model = self._model
        design = features(history, model["mean"], model["scale"],
                          model["projection"], model["bias"])
        targets = (design @ model["weights"]).reshape(CHUNK_SIZE, -1)
        targets[:, :STATE_DIM] += history[0, -STATE_DIM:]
        targets = np.clip(targets, model["low"], model["high"])

        action = {"meta.token": self._token}
        start = 0
        for key, width in zip(ACTION_KEYS, (*CHANNELS.values(), 7)):
            action[key] = np.ascontiguousarray(targets[:, start:start + width])
            start += width
        return action

    def reset(self):
        return {"ok": True}


def train(dataset, checkpoint):
    import pyarrow.parquet as parquet

    paths = sorted(dataset.glob("data/*/*.parquet"))
    if not paths:
        raise ValueError(f"No demonstration Parquet files in {dataset}")
    rng = np.random.default_rng(42)
    train_history, train_targets = [], []
    val_history, val_targets = [], []

    for path in paths:
        table = parquet.read_table(
            path, columns=[*STATE_KEYS, *ACTION_KEYS, "episode_index", "frame_index"]
        )
        episode_ids = table["episode_index"].to_numpy()
        frames = table["frame_index"].to_numpy()
        state, action = (
            np.concatenate(
                [np.asarray(table[key].to_pylist(), dtype=np.float32).reshape(len(table), -1)
                 for key in keys],
                axis=1,
            )
            for keys in (STATE_KEYS, ACTION_KEYS)
        )
        for episode in np.unique(episode_ids):
            rows = np.flatnonzero(episode_ids == episode)
            rows = rows[np.argsort(frames[rows])]
            if not np.array_equal(frames[rows], np.arange(len(rows))):
                raise ValueError(f"Episode {episode} has missing or repeated frames")
            step = np.arange(len(rows))
            history = state[rows[np.maximum(step[:, None] + OFFSETS, 0)]]
            targets = action[rows[np.minimum(step[:, None] + np.arange(CHUNK_SIZE),
                                             len(rows) - 1)]].copy()
            targets[:, :, :STATE_DIM] -= history[:, -1:, :]
            histories, actions = (
                (val_history, val_targets) if episode % 10 == 0
                else (train_history, train_targets)
            )
            histories.append(history.reshape(len(rows), -1))
            actions.append(targets)

    x, y = np.concatenate(train_history), np.concatenate(train_targets)
    val_x, val_y = np.concatenate(val_history), np.concatenate(val_targets)
    if not all(np.isfinite(array).all() for array in (x, y, val_x, val_y)):
        raise ValueError("Demonstrations contain non-finite observations or actions")
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 0.02)
    projection = rng.normal(0, 0.25 / np.sqrt(x.shape[1]), (x.shape[1], 512)).astype(np.float32)
    bias = rng.uniform(0, 2 * np.pi, 512).astype(np.float32)
    width = x.shape[1] + len(bias) + 1
    gram = np.eye(width, dtype=np.float64) * 0.1
    cross = np.zeros((width, CHUNK_SIZE * y.shape[-1]), dtype=np.float64)
    for start in range(0, len(x), 4096):
        batch = slice(start, start + 4096)
        design = features(x[batch], mean, scale, projection, bias).astype(np.float64)
        gram += design.T @ design
        cross += design.T @ y[batch].reshape(len(design), -1)
    weights = np.linalg.solve(gram, cross).astype(np.float32)
    absolute = y.copy()
    absolute[:, :, :STATE_DIM] += x[:, None, -STATE_DIM:]
    low, high = absolute.min(axis=(0, 1)), absolute.max(axis=(0, 1))
    predicted = (features(val_x, mean, scale, projection, bias) @ weights).reshape(val_y.shape)
    reference = np.zeros_like(val_y)
    reference[:, :, STATE_DIM:] = y[:, :, STATE_DIM:].mean(axis=(0, 1))
    for label, channel in (("arms", slice(0, 14)), ("right gripper", slice(15, 16))):
        mae = np.abs(predicted[:, :, channel] - val_y[:, :, channel]).mean()
        baseline = np.abs(reference[:, :, channel] - val_y[:, :, channel]).mean()
        print(f"Held-out {label} MAE: {mae:.4f} (hold baseline {baseline:.4f})")
    print(f"Episodes: {len(train_history)} train, {len(val_history)} validation")
    np.savez_compressed(checkpoint, mean=mean, scale=scale, projection=projection,
                        bias=bias, weights=weights, low=low, high=high)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    train(args.dataset, args.checkpoint)
