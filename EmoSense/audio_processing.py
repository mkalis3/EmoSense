"""
Audio processing pipeline for EmoSense.

Handles real-time audio capture, voice activity detection, speech-to-text
via Google STT, speaker diarization, and emotion analysis dispatching.
"""

import io
import gc
import time
import traceback

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import speech_recognition as sr
import tkinter as tk

import config
from utils import contains_predominantly_hebrew

from config import (
    stt_executor, stt_futures, segments_data, _last_significant_emotion_cache_spk,
    _current_1s_emotion_cache_spk, segment_id_map, PEAK_SR, CHUNK_SEC,
    SILERO_VAD_THRESHOLD, MAX_SPEECH_SEGMENT_DURATION_S, MAX_GUI_SPK,
    emotion_executor, emotion_futures, PYTORCH_AVAILABLE, ENERGY_THRESHOLD
)


def process_google_stt_task(audio_data_16k_np, google_recognizer_instance):
    if not google_recognizer_instance: return "(STT Recognizer Error)", "N/A"
    try:
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, audio_data_16k_np, 16000, format="WAV", subtype="PCM_16")
            wav_buffer.seek(0)
            with sr.AudioFile(wav_buffer) as source:
                audio_for_google = google_recognizer_instance.record(source)
    except Exception as e:
        return f"(STT Audio Prep Error: {e})", "N/A"
    try:
        text_en = google_recognizer_instance.recognize_google(audio_for_google, language="en-US")
        if text_en and not contains_predominantly_hebrew(text_en):
            return text_en.strip(), "en-US"
    except (sr.UnknownValueError, sr.RequestError):
        pass
    try:
        text_he = google_recognizer_instance.recognize_google(audio_for_google, language="he-IL")
        return text_he.strip(), "he-IL"
    except (sr.UnknownValueError, sr.RequestError):
        return "", "N/A (Unknown)"


def run_full_analysis_task(audio_segment_44k, text_result, chunk_ids, weights, is_internal_audio, speaker_id):
    from analysis import perform_full_emotion_analysis
    segment_id = chunk_ids[0] if chunk_ids else None
    emo, details, conf = perform_full_emotion_analysis(
        audio_segment=audio_segment_44k, text_segment=text_result, weights=weights,
        is_internal_audio=is_internal_audio, speaker_id=speaker_id, segment_id=segment_id
    )
    return emo, details, conf, chunk_ids


def cleanup_old_data():
    try:
        max_segments = int(600 / CHUNK_SEC)
        while len(segments_data) > max_segments:
            old_segment = segments_data.popleft()
            if old_segment['id'] in segment_id_map: del segment_id_map[old_segment['id']]
        gc.collect()
    except Exception as e:
        print(f"[Cleanup Error] {e}")


def audio_loop(stt_recognizer, log_widget):
    import diarization

    chunk_id_counter, is_speaking = 0, False
    speech_accumulator_16k, speech_accumulator_44k, speech_chunk_ids = [], [], []
    speech_start_time, current_utterance_speaker_id = 0, -1

    while True:
        try:
            completed_stt_indices = [i for i, (f, *_) in enumerate(stt_futures) if f.done()]
            for i in sorted(completed_stt_indices, reverse=True):
                future, timestamp, dom_sid, chunk_ids, audio_segment_44k, _ = stt_futures.pop(i)
                try:
                    text_result, lang = future.result()
                    if text_result and log_widget and log_widget.winfo_exists():
                        log_line = f"[{time.strftime('%H:%M:%S', time.localtime(timestamp))}] [Spk {dom_sid}]> {text_result}\n"
                        log_widget.config(state=tk.NORMAL);
                        log_widget.insert(tk.END, log_line);
                        log_widget.see(tk.END);
                        log_widget.config(state=tk.DISABLED)

                    words = text_result.split()
                    if (num_chunks := len(chunk_ids)) > 0:
                        words_per_chunk = len(words) / num_chunks
                        for j, seg_id in enumerate(chunk_ids):
                            if seg_id in segment_id_map:
                                start_idx, end_idx = int(j * words_per_chunk), int((j + 1) * words_per_chunk)
                                segment_id_map[seg_id]['words'] = words[start_idx:end_idx]

                    is_internal_audio = config.audio_source_var.get() == config.AUDIO_SOURCE_INTERNAL
                    current_weights = {k: v.get() / 100.0 for k, v in config.emotion_weight_vars.items()}
                    emo_future = emotion_executor.submit(run_full_analysis_task, audio_segment_44k, text_result,
                                                         chunk_ids, current_weights, is_internal_audio, dom_sid)
                    emotion_futures.append(emo_future)
                except Exception as e:
                    print(f"Error processing STT future: {e}", flush=True)

            completed_emo_indices = [i for i, f in enumerate(emotion_futures) if f.done()]
            for i in sorted(completed_emo_indices, reverse=True):
                future = emotion_futures.pop(i)
                try:
                    if result := future.result():
                        emo, details, conf, chunk_ids = result

                        if chunk_ids and chunk_ids[0] % 10 == 0:
                            print(f"[AUDIO DEBUG] Storing emotion details for segments {chunk_ids}: {emo} ({conf:.0%})")

                        for seg_id in chunk_ids:
                            if seg_id in segment_id_map:
                                segment_id_map[seg_id]['emotion_state'] = emo
                                segment_id_map[seg_id]['emotion_confidence'] = conf
                                segment_id_map[seg_id]['emotion_details'] = details

                                if seg_id % 10 == 0:
                                    stored_details = segment_id_map[seg_id].get('emotion_details', {})
                                    print(f"[AUDIO DEBUG] Verified storage for segment {seg_id}:")
                                    print(f"  - Has CNN: {'cnn_analysis' in stored_details}")
                                    print(f"  - Has Logic: {'logic_analysis' in stored_details}")
                                    print(f"  - Has Text: {'text_analysis' in stored_details}")
                                    print(f"  - Has Final: {'final_decision' in stored_details}")

                        if chunk_ids and 0 <= (dom_sid := segment_id_map[chunk_ids[0]].get("sid", -1)) < MAX_GUI_SPK:
                            if emo not in ["neutral", "silent"] or conf > 0.6:
                                _last_significant_emotion_cache_spk[dom_sid] = {'emotion': emo, 'time': time.time(),
                                                                                'confidence': conf}
                except Exception as e:
                    print(f"Error processing emotion future: {e}", flush=True)
                    traceback.print_exc()

            block_size = int(CHUNK_SEC * PEAK_SR)
            if config.current_audio_device_index != config.audio_loop_last_dev_idx_val or not \
                    config.audio_loop_stream_obj_ref[0]:
                if config.audio_loop_stream_obj_ref[0]:
                    try:
                        config.audio_loop_stream_obj_ref[0].close()
                    except Exception:
                        pass
                try:
                    dev = config.current_audio_device_index if config.current_audio_device_index != -1 else None
                    config.audio_loop_stream_obj_ref[0] = sd.InputStream(samplerate=PEAK_SR, device=dev, channels=1,
                                                                         dtype='float32', blocksize=block_size)
                    config.audio_loop_stream_obj_ref[0].start()
                    config.audio_loop_last_dev_idx_val = config.current_audio_device_index
                except Exception:
                    config.audio_loop_stream_obj_ref[0] = None;
                    time.sleep(1);
                    continue

            y_data, _ = config.audio_loop_stream_obj_ref[0].read(block_size)
            y_data = y_data.flatten()

            has_speech = False
            if PYTORCH_AVAILABLE and config.vad_model and config.get_speech_timestamps:
                try:
                    vad_input = librosa.resample(y_data, orig_sr=PEAK_SR, target_sr=16000)
                    vad_input_tensor = config.torch.from_numpy(vad_input)
                    speech_timestamps = config.get_speech_timestamps(vad_input_tensor, config.vad_model,
                                                                     threshold=SILERO_VAD_THRESHOLD)
                    if speech_timestamps: has_speech = True
                except Exception as e:
                    print(f"[VAD ERROR] {e}, falling back to energy.", flush=True)
                    has_speech = np.sqrt(np.mean(y_data ** 2)) > ENERGY_THRESHOLD
            else:
                has_speech = np.sqrt(np.mean(y_data ** 2)) > ENERGY_THRESHOLD

            if has_speech:
                if not is_speaking:
                    is_speaking, speech_start_time = True, time.time()
                    speech_accumulator_16k.clear();
                    speech_accumulator_44k.clear();
                    speech_chunk_ids.clear()
                    try:
                        emb = config.speaker_encoder.embed_utterance(
                            librosa.resample(y_data, orig_sr=PEAK_SR, target_sr=16000))
                        int_sid = diarization.dia.add(emb)
                        current_utterance_speaker_id = config.gui_speaker_mapper.get_gui_sid(int_sid, emb)
                    except Exception as e:
                        print(f"Diarization error: {e}", flush=True);
                        current_utterance_speaker_id = 0

                speech_accumulator_16k.append(librosa.resample(y_data, orig_sr=PEAK_SR, target_sr=16000))
                speech_accumulator_44k.append(y_data)
                speech_chunk_ids.append(chunk_id_counter)

                segment_info = {"id": chunk_id_counter, "audio": y_data, "words": [],
                                "sid": current_utterance_speaker_id, "emotion_state": "__processing__"}
                segment_id_map[chunk_id_counter] = segment_info
                segments_data.append(segment_info)

                if 0 <= current_utterance_speaker_id < MAX_GUI_SPK:
                    _current_1s_emotion_cache_spk[current_utterance_speaker_id]["text"] = "Speaking..."
            else:
                for i in range(MAX_GUI_SPK):
                    if i != current_utterance_speaker_id or not is_speaking:
                        _current_1s_emotion_cache_spk[i]["text"] = "Silent"

            end_of_utterance = (not has_speech and is_speaking and speech_accumulator_16k) or (
                    is_speaking and time.time() - speech_start_time >= MAX_SPEECH_SEGMENT_DURATION_S)
            if end_of_utterance:
                future = stt_executor.submit(process_google_stt_task, np.concatenate(speech_accumulator_16k),
                                             stt_recognizer)
                stt_futures.append((future, time.time(), current_utterance_speaker_id, list(speech_chunk_ids),
                                    np.concatenate(speech_accumulator_44k), None))
                is_speaking, current_utterance_speaker_id = False, -1
                speech_accumulator_16k.clear();
                speech_chunk_ids.clear();
                speech_accumulator_44k.clear()

            chunk_id_counter += 1
            if chunk_id_counter % 300 == 0: cleanup_old_data()
            time.sleep(0.005)

        except sd.PortAudioError as pa_err:
            print(f"PortAudioError, resetting stream: {pa_err}", flush=True)
            if config.audio_loop_stream_obj_ref[0]: config.audio_loop_stream_obj_ref[0].close()
            config.audio_loop_stream_obj_ref[0] = None;
            time.sleep(1)
        except Exception as e:
            print(f"AudioLoop Critical Error: {e}", flush=True);
            traceback.print_exc()
            if config.audio_loop_stream_obj_ref[0]: config.audio_loop_stream_obj_ref[0].close()
            config.audio_loop_stream_obj_ref[0] = None;
            time.sleep(1)