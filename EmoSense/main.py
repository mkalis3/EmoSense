"""EmoSense application entry point."""

import os
import sys
import warnings
import threading
import tkinter as tk
from tkinter import messagebox
import time


def heavy_initialization_task(progress_queue, loading_complete_event):
    try:
        import tensorflow as tf
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        from resemblyzer import VoiceEncoder
        from transformers import pipeline
        import config
        import diarization

        tf.get_logger().setLevel('ERROR')

        start_time = time.time()
        min_display_time = getattr(config, 'LOADING_MIN_TIME', 4.0)

        progress_queue.put((5, "Loading models..."))

        if os.path.exists(config.CNN_PATH):
            config.cnn_model = tf.keras.models.load_model(config.CNN_PATH, compile=False)
            if config.cnn_model is not None:
                print("[Init] CNN model loaded successfully.")
            else:
                print("[Init] ERROR: CNN model failed to load.")

            if os.path.exists(config.LABEL_PATH):
                config.cnn_lblenc = LabelEncoder()
                config.cnn_lblenc.classes_ = np.load(config.LABEL_PATH, allow_pickle=True)
                print("[Init] Label encoder loaded.")
        else:
            print(f"[Init] ERROR: CNN model file not found at {config.CNN_PATH}")

        progress_queue.put((20, "CNN model loaded"))

        config.speaker_encoder = VoiceEncoder()
        print("[Init] Voice encoder loaded.")
        progress_queue.put((35, "Voice encoder loaded"))

        if config.PYTORCH_AVAILABLE:
            import torch
            config.torch = torch
            try:
                config.vad_model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True, onnx=True
                )
                config.get_speech_timestamps = utils[0]
                print("[Init] VAD model loaded.")
            except Exception as e:
                print(f"[Init] VAD Loading Failed: {e}")
        progress_queue.put((50, "VAD loaded"))

        config.text_emotion_pipeline = pipeline(
            "text-classification", model="cardiffnlp/twitter-roberta-base-emotion", top_k=None
        )
        print("[Init] Text analysis pipeline loaded.")
        progress_queue.put((65, "Text analysis loaded"))

        try:
            config.hebrew_sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="avichr/heBERT_sentiment_analysis",
                device=-1
            )
            print("[Init] Hebrew sentiment model loaded successfully.")
        except Exception as e:
            print(f"[Init] Hebrew model loading failed (optional): {e}")
            config.hebrew_sentiment_pipeline = None
        progress_queue.put((80, "Language models loaded"))

        diarization.dia = diarization.OnlineDia()
        config.gui_speaker_mapper = diarization.GuiSpeakerMapper()
        print("[Init] Diarization module initialized.")
        progress_queue.put((90, "Diarization initialized"))

        elapsed = time.time() - start_time
        if elapsed < min_display_time:
            time.sleep(min_display_time - elapsed)

        progress_queue.put((100, "Initialization complete!"))
        time.sleep(1)

    except Exception as e:
        error_msg = f"Critical loading error: {str(e)}"
        progress_queue.put((98, error_msg))
        print(f"FATAL ERROR DURING INITIALIZATION: {e}")
        import traceback
        traceback.print_exc()
    finally:
        loading_complete_event.set()


def start_app():
    root = tk.Tk()
    root.withdraw()

    import config
    import gui

    loading_complete_event = threading.Event()

    init_thread = threading.Thread(
        target=heavy_initialization_task,
        args=(config.loading_progress_queue, loading_complete_event),
        daemon=True
    )
    init_thread.start()

    gui.show_loading_screen(root, loading_complete_event)

    def on_close():
        if hasattr(root, 'generate_report') and messagebox.askyesno("Exit", "Generate report before closing?",
                                                                    parent=root):
            root.generate_report()
        if hasattr(root, 'save_settings'):
            root.save_settings()
        root.destroy()
        sys.exit()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    start_app()