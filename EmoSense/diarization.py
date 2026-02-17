"""
Speaker diarization module for EmoSense.

Provides online speaker identification using cosine similarity on voice
embeddings, GUI slot mapping, and optional acoustic feature extraction.
"""

import numpy as np
import time
import config

try:
    from scipy.spatial.distance import cdist

    HAS_SCIPY = True
except ImportError:
    print("WARNING: scipy not installed - using basic distance calculation")
    HAS_SCIPY = False


    def cdist(a, b, metric='cosine'):
        if metric == 'cosine':
            dot_product = np.dot(a[0], b[0])
            norm_a = np.linalg.norm(a[0])
            norm_b = np.linalg.norm(b[0])
            if norm_a == 0 or norm_b == 0:
                return np.array([[1.0]])
            return np.array([[1 - dot_product / (norm_a * norm_b)]])
        return np.array([[np.linalg.norm(a[0] - b[0])]])

try:
    from sklearn.mixture import GaussianMixture

    HAS_SKLEARN = True
except ImportError:
    print("WARNING: sklearn not installed - GMM features disabled")
    HAS_SKLEARN = False
    GaussianMixture = None

try:
    import librosa
    _ = librosa  # verify availability

    HAS_LIBROSA = True
except ImportError:
    print("WARNING: Enhanced acoustic features disabled")
    HAS_LIBROSA = False


class OnlineDia:
    """Online speaker diarization using cosine similarity on voice embeddings."""

    def __init__(self, momentum=config.CENTROID_MOMENTUM, max_inactive_time=config.DIA_MAX_INACTIVE_TIME):
        self.momentum = momentum
        self.max_inactive_time = max_inactive_time
        self.centroids = []
        self.speaker_ids = []
        self.next_speaker_id = 0
        self.last_seen = []
        self.fixed_threshold = 0.9

        self.speakers = {}
        self.feature_extractor = None
        if HAS_LIBROSA:
            self.feature_extractor = AcousticFeatureExtractor()

    def add(self, emb_or_audio, embedding=None, duration=1.0):
        """Assign a speaker ID to an embedding, creating new speakers as needed."""
        if embedding is None:
            embedding = emb_or_audio
        else:
            _ = emb_or_audio  # audio_chunk reserved for future use

        current_threshold = 0.9

        if not self.centroids:
            self._add_new_speaker(embedding)
            return self.speaker_ids[-1]

        distances = cdist([embedding], self.centroids, metric='cosine').flatten()
        best_match_idx = np.argmin(distances)
        best_sim = 1 - distances[best_match_idx]

        if best_sim < current_threshold:
            self._add_new_speaker(embedding)
            return self.speaker_ids[-1]
        else:
            self.centroids[best_match_idx] = (
                    self.momentum * self.centroids[best_match_idx] +
                    (1 - self.momentum) * embedding
            )
            self.last_seen[best_match_idx] = time.time()
            return self.speaker_ids[best_match_idx]

    def _add_new_speaker(self, emb):
        self.centroids.append(emb)
        self.speaker_ids.append(self.next_speaker_id)
        self.last_seen.append(time.time())
        self.next_speaker_id += 1

    def prune_speakers(self):
        now = time.time()
        to_keep = [i for i, last_t in enumerate(self.last_seen)
                   if now - last_t < self.max_inactive_time]
        self.centroids = [self.centroids[i] for i in to_keep]
        self.speaker_ids = [self.speaker_ids[i] for i in to_keep]
        self.last_seen = [self.last_seen[i] for i in to_keep]


class GuiSpeakerMapper:

    def __init__(self, num_gui_slots=config.MAX_GUI_SPK):
        self.num_gui_slots = num_gui_slots
        self.internal_to_gui = {}
        self.gui_to_internal = {}
        self.gui_slots_last_seen = [-1] * num_gui_slots
        self.gui_slots_embeddings = [None] * num_gui_slots

    def get_gui_sid(self, internal_sid, emb, profile=None):
        if internal_sid in self.internal_to_gui:
            gui_sid = self.internal_to_gui[internal_sid]
            self.gui_slots_last_seen[gui_sid] = time.time()
            return gui_sid

        now = time.time()
        inactive_times = [now - t if t != -1 else float('inf')
                          for t in self.gui_slots_last_seen]

        if any(t > config.RESET_TO_NEUTRAL_AFTER for t in inactive_times):
            gui_sid = np.argmax(inactive_times)
        else:
            gui_sid = np.argmin(self.gui_slots_last_seen) if any(
                t != -1 for t in self.gui_slots_last_seen) else 0

        if self.gui_slots_last_seen[gui_sid] != -1:
            old_internal_sid = self.gui_to_internal.get(gui_sid)
            if old_internal_sid is not None and old_internal_sid in self.internal_to_gui:
                del self.internal_to_gui[old_internal_sid]

        self.internal_to_gui[internal_sid] = gui_sid
        self.gui_to_internal[gui_sid] = internal_sid
        self.gui_slots_last_seen[gui_sid] = now
        self.gui_slots_embeddings[gui_sid] = emb

        return gui_sid

    def get_speaker_info(self, gui_sid):
        return None


if HAS_LIBROSA:
    class AcousticFeatureExtractor:

        def __init__(self, sr=16000):
            self.sr = sr

        def extract_features(self, audio_chunk):
            try:
                rms = np.sqrt(np.mean(audio_chunk ** 2))
                zcr = np.mean(np.diff(np.sign(audio_chunk)) != 0)
                return np.array([rms, zcr])
            except Exception:
                return np.array([0, 0])

EnhancedOnlineDia = OnlineDia

dia = None
gui_speaker_mapper = None

print("Diarization module loaded")
print(f"- scipy available: {HAS_SCIPY}")
print(f"- sklearn available: {HAS_SKLEARN}")
print(f"- librosa features: {HAS_LIBROSA}")