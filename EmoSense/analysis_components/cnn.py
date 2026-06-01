"""CNN emotion inference."""

import librosa
import numpy as np

import config

def cnn_emotion_analysis(audio_segment, segment_id=None):
    """Run CNN model inference on an audio segment.

    Extracts MFCCs, pads/truncates to expected dimensions, runs prediction,
    and applies neutrality injection and anger dampening.

    Args:
        audio_segment: NumPy array of audio samples.
        segment_id: Optional identifier for logging.

    Returns:
        Tuple of (probability_distribution, reason_string).
    """
    if not config.cnn_model or not config.cnn_lblenc or audio_segment is None:
        return {}, "N/A"
    try:
        mfccs = librosa.feature.mfcc(y=audio_segment, sr=config.PEAK_SR, n_mfcc=config.CNN_EXPECTED_MFCC_FEATURES)
        if mfccs.shape[1] > config.CNN_EXPECTED_TIME_STEPS:
            mfccs = mfccs[:, :config.CNN_EXPECTED_TIME_STEPS]
        else:
            mfccs = np.pad(mfccs, ((0, 0), (0, config.CNN_EXPECTED_TIME_STEPS - mfccs.shape[1])), 'constant')

        raw_probs_array = config.cnn_model.predict(mfccs[np.newaxis, ..., np.newaxis], verbose=0)[0]
        raw_dist = {config.cnn_lblenc.classes_[i].lower(): prob for i, prob in enumerate(raw_probs_array)}

        injected_dist = raw_dist.copy()

        injected_dist['neutral'] = injected_dist.get('neutral', 0) + config.CNN_NEUTRALITY_INJECTION

        if 'angry' in injected_dist and injected_dist['angry'] > 0.3:
            anger_reduction = (injected_dist['angry'] - 0.3) * 0.5
            injected_dist['angry'] -= anger_reduction
            injected_dist['neutral'] += anger_reduction * 0.7
            injected_dist['happy'] += anger_reduction * 0.15
            injected_dist['sad'] += anger_reduction * 0.15

        total_prob = sum(injected_dist.values())
        final_dist = {k: v / total_prob for k, v in injected_dist.items()} if total_prob > 0 else {}

        top_emotion = max(final_dist, key=final_dist.get, default="N/A")
        return final_dist, f"Suggests '{top_emotion}' ({final_dist.get(top_emotion, 0) * 100:.0f}%)"
    except Exception as e:
        return {}, f"CNN Error: {e}"
