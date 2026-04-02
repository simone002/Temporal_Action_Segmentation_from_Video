import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from collections import defaultdict

FPS = 24  # EGTEA Gaze+ è girato a 24 fps


def ms_to_frame(ms: int) -> int:
    """Converte millisecondi in frame index (1-based)."""
    return max(1, int(ms / 1000 * FPS))


def load_action_labels(csv_path: str) -> dict:
    """
    action_labels.csv + cls_label_index.csv
    → {video_session: [(frame_start, frame_end, action_id), ...]}
    action_id è 0-indexed (da cls_label_index.csv).
    """
    cls_path = csv_path.replace("action_labels.csv", "cls_label_index.csv")
    label_to_id = {}
    with open(cls_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) >= 2:
                try:
                    label_to_id[parts[1].strip()] = int(parts[0])
                except ValueError:
                    continue

    annotations = defaultdict(list)
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) < 6:
                continue
            try:
                video_session = parts[2].strip()
                start_ms      = int(parts[3])
                end_ms        = int(parts[4])
                action_label  = parts[5].strip()
                action_id     = label_to_id.get(action_label, -1)
                if action_id == -1:
                    continue
                annotations[video_session].append(
                    (ms_to_frame(start_ms), ms_to_frame(end_ms), action_id)
                )
            except (ValueError, IndexError):
                continue

    return dict(annotations)


def load_split(split_file: str) -> list:
    clips = []
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            clip_prefix   = parts[0]
            tokens        = clip_prefix.split("-")
            frame_start   = int(tokens[-2][1:])
            frame_end     = int(tokens[-1][1:])
            video_session = "-".join(tokens[:-4])
            clips.append({
                "clip_prefix":   clip_prefix,
                "video_session": video_session,
                "frame_start":   frame_start,
                "frame_end":     frame_end,
                "action_id":     int(parts[1]) - 1,
                "verb_id":       int(parts[2]) - 1,
                "noun_id":       int(parts[3]) - 1,
            })
    return clips


_LMDB_ENVS: dict = {}


class LMDBReader:
    def __init__(self, path: str, feat_dim: int = 1024):
        self.path     = path
        self.feat_dim = feat_dim

    def _get_env(self):
        if self.path not in _LMDB_ENVS:
            _LMDB_ENVS[self.path] = lmdb.open(
                self.path, readonly=True, lock=False,
                readahead=False, meminit=False,
            )
        return _LMDB_ENVS[self.path]

    def get_frame(self, video_session: str, frame_idx: int):
        key = f"{video_session}_frame_{frame_idx:010d}.jpg"
        with self._get_env().begin(write=False) as txn:
            data = txn.get(key.encode("utf-8"))
        if data is None:
            return None
        return np.frombuffer(data, dtype=np.float32).copy()

    def get_clip(self, video_session: str, frame_start: int, frame_end: int):
        frames = []
        for idx in range(frame_start, frame_end + 1):
            feat = self.get_frame(video_session, idx)
            if feat is None:
                feat = np.zeros(self.feat_dim, dtype=np.float32)
            frames.append(feat)
        return np.stack(frames, axis=0)


class EGTEADataset(Dataset):
    """
    Dataset con label dense frame-per-frame.
    Ogni frame riceve la label dell'azione che lo copre (0 = background).
    """

    def __init__(
        self,
        lmdb_rgb_path: str,
        annotation_dir: str,
        split_file: str,
        seq_len: int = 256,
        feat_dim: int = 1024,
        use_flow: bool = False,
        lmdb_flow_path=None,
        background_id: int = 0,
    ):
        self.seq_len       = seq_len
        self.feat_dim      = feat_dim
        self.use_flow      = use_flow
        self.background_id = background_id

        ann_dir = Path(annotation_dir)
        self.clips = load_split(str(ann_dir / split_file))

        # Annotazioni dense da raw_annotations/
        raw_ann = ann_dir / "raw_annotations" / "action_labels.csv"
        self.dense_annotations = load_action_labels(str(raw_ann))

        self.rgb_reader  = LMDBReader(lmdb_rgb_path, feat_dim=feat_dim)
        self.flow_reader = LMDBReader(lmdb_flow_path, feat_dim=feat_dim) \
                           if use_flow and lmdb_flow_path else None

    def __len__(self):
        return len(self.clips)

    def _build_dense_labels(self, video_session, frame_start, frame_end):
        T      = frame_end - frame_start + 1
        labels = np.full(T, self.background_id, dtype=np.int64)
        for (ann_start, ann_end, action_id) in self.dense_annotations.get(video_session, []):
            o_start = max(ann_start, frame_start)
            o_end   = min(ann_end,   frame_end)
            if o_start <= o_end:
                labels[o_start - frame_start : o_end - frame_start + 1] = action_id
        return labels

    def __getitem__(self, idx):
        clip = self.clips[idx]

        feat = self.rgb_reader.get_clip(
            clip["video_session"], clip["frame_start"], clip["frame_end"]
        )

        if self.use_flow and self.flow_reader is not None:
            flow = self.flow_reader.get_clip(
                clip["video_session"], clip["frame_start"], clip["frame_end"]
            )
            feat = np.concatenate([feat, flow], axis=-1)

        labels = self._build_dense_labels(
            clip["video_session"], clip["frame_start"], clip["frame_end"]
        )

        feat, labels = self._pad_or_crop(feat, labels)

        return torch.from_numpy(feat).float(), torch.from_numpy(labels).long()

    def _pad_or_crop(self, feat, labels):
        T = feat.shape[0]
        if T >= self.seq_len:
            start  = np.random.randint(0, T - self.seq_len + 1)
            feat   = feat[start : start + self.seq_len]
            labels = labels[start : start + self.seq_len]
        else:
            pad = self.seq_len - T
            feat   = np.concatenate([feat,   np.zeros((pad, feat.shape[1]), dtype=np.float32)])
            labels = np.concatenate([labels, np.full(pad, self.background_id, dtype=np.int64)])
        return feat, labels