"""
GUI module for EmoSense.

Provides the Tkinter-based interface with real-time waveform display,
emotion analysis panels, audio source selection, and report generation.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from queue import Empty
import threading
import time

import config
from utils import save_settings, load_settings
import report_generator


def show_loading_screen(root_window, loading_complete_event):
    loading_win = tk.Toplevel(root_window)
    loading_win.title("Loading...")
    loading_win.geometry("500x250")
    loading_win.resizable(False, False)
    loading_win.overrideredirect(True)

    loading_win.update_idletasks()
    x = (loading_win.winfo_screenwidth() // 2) - 250
    y = (loading_win.winfo_screenheight() // 2) - 125
    loading_win.geometry(f"500x250+{x}+{y}")

    main_frame = ttk.Frame(loading_win, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(main_frame, text="EMOSENSE", font=("Arial", 20, "bold")).pack(pady=10)
    status_label = ttk.Label(main_frame, text="Initializing...", font=("Arial", 12))
    status_label.pack(pady=10)
    progress_bar = ttk.Progressbar(main_frame, orient='horizontal', length=400, mode='determinate')
    progress_bar.pack(pady=10)
    percent_label = ttk.Label(main_frame, text="0%", font=("Arial", 10, "bold"))
    percent_label.pack()

    loading_win.lift()

    def check_for_completion():
        try:
            progress, status = config.loading_progress_queue.get_nowait()
            progress_bar['value'] = progress
            percent_label.config(text=f"{int(progress)}%")
            status_label.config(text=status)
            loading_win.update_idletasks()
        except Empty:
            pass

        if loading_complete_event.is_set():
            loading_win.after(500, lambda: (loading_win.destroy(), build_main_gui(root_window)))
        else:
            loading_win.after(100, check_for_completion)

    loading_win.after(100, check_for_completion)


def add_scroll_slider(fig, on_scroll_update, SliderWidget):
    ax = fig.add_axes([0.15, 0.01, 0.7, 0.03])
    s = SliderWidget(ax, 'Time', 0, 1.0, valinit=0, valstep=0.2, color='lightblue')
    s.on_changed(on_scroll_update)
    return s


def populate_internal_audio_devices(cb, sd):
    if not cb or not cb.winfo_exists(): return
    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"Could not query audio devices: {e}")
        devices = []
    cb['values'] = [f"{i}: {d.get('name', 'Unk')} ({sd.query_hostapis(d['hostapi']).get('name', 'N/A')})"
                    for i, d in enumerate(devices) if d.get('max_input_channels', 0) > 0]
    if (saved_name := config.internal_audio_device_var.get()) in cb['values']:
        cb.set(saved_name)
    elif cb['values']:
        cb.set(cb['values'][0])
        config.internal_audio_device_var.set(cb['values'][0])


def update_audio_source(sd):
    try:
        if config.audio_source_var.get() == config.AUDIO_SOURCE_INTERNAL:
            config.current_audio_device_index = int(config.internal_audio_device_var.get().split(':')[0])
        else:
            config.current_audio_device_index = sd.default.device[0]
    except (ValueError, IndexError, AttributeError):
        config.current_audio_device_index = -1


def show_segment_details(segment_id, parent_window):
    if not segment_id or segment_id not in config.segment_id_map or 'emotion_details' not in config.segment_id_map.get(
            segment_id, {}):
        messagebox.showinfo("No Details", "Analysis details are not yet available for this segment.",
                            parent=parent_window)
        return
    details = config.segment_id_map[segment_id]['emotion_details']
    win = tk.Toplevel(parent_window)
    win.title(f"Analysis Details - Segment {segment_id}")
    win.geometry("600x550")
    win.resizable(False, False)
    win.transient(parent_window)
    win.grab_set()
    main_frame = ttk.Frame(win, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)

    final_frame = ttk.LabelFrame(main_frame, text="Final Emotion Decision", padding=10)
    final_frame.pack(fill=tk.X, pady=5)
    final_emo = details.get('final_decision', 'N/A').capitalize()
    final_conf = details.get('final_confidence', 0) * 100
    ttk.Label(final_frame, text=f"{final_emo} ({final_conf:.1f}%)",
              font=("Helvetica", 14, "bold")).pack()

    text_frame = ttk.LabelFrame(main_frame, text="Transcribed Text", padding=10)
    text_frame.pack(fill=tk.X, pady=5)
    raw_text = details.get('raw_text', 'N/A')
    if any('\u0590' <= c <= '\u05FF' for c in raw_text):
        ttk.Label(text_frame, text=raw_text, wraplength=550, justify=tk.RIGHT).pack(anchor='e')
    else:
        ttk.Label(text_frame, text=raw_text, wraplength=550, justify=tk.LEFT).pack(anchor='w')

    spam_frame = ttk.LabelFrame(main_frame, text="Spam Detection", padding=10)
    spam_frame.pack(fill=tk.X, pady=5)
    spam_info = details.get('spam_detection', {})
    spam_confidence = spam_info.get('confidence', 0)
    spam_text = f"IDENTIFYING SPAM: {spam_confidence * 100:.0f}%"
    spam_color = "red" if spam_confidence > 0.5 else "black"
    ttk.Label(spam_frame, text=spam_text, foreground=spam_color, font=("Helvetica", 10, "bold")).pack()
    spam_reason = spam_info.get('reason', 'Normal conversation pattern')
    ttk.Label(spam_frame, text=f"Reason: {spam_reason}", wraplength=550).pack()

    breakdown_frame = ttk.LabelFrame(main_frame, text="Emotion Model Breakdown", padding=10)
    breakdown_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    def create_detail_row(parent, model_name, data, weight):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=4)
        ttk.Label(frame, text=f"{model_name}:", font=("Helvetica", 10, "bold"), width=7).pack(side=tk.LEFT, anchor='n')
        ttk.Label(frame, text=f"({weight * 100:.0f}% w)", foreground="grey", font=("Helvetica", 8)).pack(side=tk.LEFT,
                                                                                                         anchor='n',
                                                                                                         padx=5)
        info_frame = ttk.Frame(frame)
        info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        dist_text = ", ".join([f"{k.capitalize()}: {v * 100:.1f}%" for k, v in
                               sorted(data.get('dist', {}).items(), key=lambda item: item[1], reverse=True) if
                               v > 0.01])
        ttk.Label(info_frame, text=data.get('reason', 'N/A'), wraplength=400, justify=tk.LEFT).pack(anchor='w')
        if dist_text: ttk.Label(info_frame, text=f"Scores: {dist_text}", foreground="darkblue",
                                font=("Helvetica", 8, "italic")).pack(anchor='w')

    weights = details.get('final_weights', {})
    create_detail_row(breakdown_frame, "CNN", details.get('cnn_analysis', {}), weights.get('cnn', 0))
    create_detail_row(breakdown_frame, "Logic", details.get('logic_analysis', {}), weights.get('logic', 0))
    create_detail_row(breakdown_frame, "Text", details.get('text_analysis', {}), weights.get('text', 0))


def build_main_gui(root_window):
    import numpy as np
    import sounddevice as sd
    import speech_recognition as sr
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Slider
    from audio_processing import audio_loop

    root_window.deiconify()
    root_window.geometry("1500x900")
    root_window.title("EMOSENSE - Real-Time Emotion Analyzer")

    saved = load_settings()
    config.audio_source_var = tk.StringVar(root_window, value=saved.get('audio_source', config.AUDIO_SOURCE_EXTERNAL))
    config.internal_audio_device_var = tk.StringVar(root_window, value=saved.get('internal_device_name', ''))
    config.emotion_weight_vars = {
        k: tk.DoubleVar(root_window, value=saved.get('weights', config.INITIAL_EMOTION_WEIGHTS).get(k, v) * 100) for
        k, v in config.INITIAL_EMOTION_WEIGHTS.items()}

    main_pane = ttk.PanedWindow(root_window, orient=tk.HORIZONTAL)
    main_pane.pack(fill=tk.BOTH, expand=True)
    speakers_frame = ttk.Frame(main_pane, padding=5)
    main_pane.add(speakers_frame, weight=2)
    right_panel = ttk.Frame(main_pane, padding=5)
    main_pane.add(right_panel, weight=1)
    log_text = tk.Text(right_panel, wrap=tk.WORD, font=("Arial", 9), state=tk.DISABLED)

    def save_settings_handler():
        total = sum(v.get() for v in config.emotion_weight_vars.values()) or 1
        w = {k: v.get() / total for k, v in config.emotion_weight_vars.items()}
        save_settings({'weights': w, 'audio_source': config.audio_source_var.get(),
                       'internal_device_name': config.internal_audio_device_var.get()})

    def reset_handler(log_widget):
        save_settings_handler()
        config.segments_data.clear()
        config.segment_id_map.clear()
        if hasattr(config, 'distress_detection_history'):
            config.distress_detection_history.clear()
        if hasattr(config, 'current_distress_status'):
            config.current_distress_status.clear()
        log_widget.config(state=tk.NORMAL)
        log_widget.delete('1.0', tk.END)
        log_widget.config(state=tk.DISABLED)
        messagebox.showinfo("Reset", "State has been reset.", parent=root_window)

    def generate_report_handler():
        report_path = report_generator.generate_summary_report()
        if report_path:
            messagebox.showinfo("Report Generated", f"Summary report saved to:\n{report_path}", parent=root_window)
        else:
            messagebox.showwarning("Report Failed", "Could not generate a report. No data available.",
                                   parent=root_window)

    reset_btn = ttk.Button(right_panel, text="Save & Reset", command=lambda: reset_handler(log_text))
    reset_btn.pack(pady=2, fill=tk.X)
    report_btn = ttk.Button(right_panel, text="Generate Report", command=generate_report_handler)
    report_btn.pack(pady=(0, 5), fill=tk.X)

    distress_frame = ttk.LabelFrame(right_panel, text="Distress Detection", padding=5)
    distress_frame.pack(fill=tk.X, pady=3)
    distress_status_label = ttk.Label(distress_frame, text="At Risk: No", font=("Arial", 11, "bold"))
    distress_status_label.pack()
    distress_detail_label = ttk.Label(distress_frame, text="Status: Monitoring...", font=("Arial", 8))
    distress_detail_label.pack()
    root_window.distress_status_label = distress_status_label
    root_window.distress_detail_label = distress_detail_label

    analysis_f = ttk.LabelFrame(right_panel, text="Last Segment Analysis", padding=5)
    analysis_f.pack(fill=tk.X, pady=3)

    overall_frame = ttk.Frame(analysis_f)
    overall_frame.pack(fill=tk.X, pady=(0, 8))

    overall_title = ttk.Label(overall_frame, text="Overall Analysis (5s)", font=("Arial", 10, "bold"))
    overall_title.pack(anchor=tk.W)

    overall_sections = ttk.Frame(overall_frame)
    overall_sections.pack(fill=tk.X, padx=10)

    model_overall_frame = ttk.Frame(overall_sections)
    model_overall_frame.pack(fill=tk.X)
    lbl_model_overall_title = ttk.Label(model_overall_frame, text="Model:", width=8)
    lbl_model_overall_title.pack(side=tk.LEFT)
    lbl_model_overall = ttk.Label(model_overall_frame, text="N/A", foreground="gray")
    lbl_model_overall.pack(side=tk.LEFT, padx=5)
    lbl_model_overall_weight = ttk.Label(model_overall_frame, text="", foreground="gray", font=("Arial", 8))
    lbl_model_overall_weight.pack(side=tk.LEFT)

    logic_overall_frame = ttk.Frame(overall_sections)
    logic_overall_frame.pack(fill=tk.X)
    lbl_logic_overall_title = ttk.Label(logic_overall_frame, text="Logic:", width=8)
    lbl_logic_overall_title.pack(side=tk.LEFT)
    lbl_logic_overall = ttk.Label(logic_overall_frame, text="N/A", foreground="gray")
    lbl_logic_overall.pack(side=tk.LEFT, padx=5)
    lbl_logic_overall_weight = ttk.Label(logic_overall_frame, text="", foreground="gray", font=("Arial", 8))
    lbl_logic_overall_weight.pack(side=tk.LEFT)

    stt_overall_frame = ttk.Frame(overall_sections)
    stt_overall_frame.pack(fill=tk.X)
    lbl_stt_overall_title = ttk.Label(stt_overall_frame, text="STT:", width=8)
    lbl_stt_overall_title.pack(side=tk.LEFT)
    lbl_stt_overall = ttk.Label(stt_overall_frame, text="N/A", foreground="gray")
    lbl_stt_overall.pack(side=tk.LEFT, padx=5)
    lbl_stt_overall_weight = ttk.Label(stt_overall_frame, text="", foreground="gray", font=("Arial", 8))
    lbl_stt_overall_weight.pack(side=tk.LEFT)

    ttk.Separator(analysis_f, orient='horizontal').pack(fill='x', pady=5)

    model_section = ttk.Frame(analysis_f)
    model_section.pack(fill=tk.X, pady=(0, 5))

    f_cnn = ttk.Frame(model_section)
    f_cnn.pack(fill=tk.X, pady=2)
    lbl_cnn_title = ttk.Label(f_cnn, text="CNN Model:", font=("Arial", 9, "bold"), width=10)
    lbl_cnn_title.pack(side=tk.LEFT)
    lbl_cnn = ttk.Label(f_cnn, text="N/A", wraplength=250, font=("Arial", 8))
    lbl_cnn.pack(side=tk.LEFT, padx=5)
    lbl_cnn_w = ttk.Label(f_cnn, text="(W: 0%)", foreground="gray", font=("Arial", 8))
    lbl_cnn_w.pack(side=tk.LEFT)

    f_logic = ttk.Frame(model_section)
    f_logic.pack(fill=tk.X, pady=2)
    lbl_logic_title = ttk.Label(f_logic, text="Logic:", font=("Arial", 9, "bold"), width=10)
    lbl_logic_title.pack(side=tk.LEFT)
    lbl_logic = ttk.Label(f_logic, text="N/A", wraplength=250, font=("Arial", 8))
    lbl_logic.pack(side=tk.LEFT, padx=5)
    lbl_logic_w = ttk.Label(f_logic, text="(W: 0%)", foreground="gray", font=("Arial", 8))
    lbl_logic_w.pack(side=tk.LEFT)

    f_text = ttk.Frame(model_section)
    f_text.pack(fill=tk.X, pady=2)
    lbl_text_title = ttk.Label(f_text, text="Text:", font=("Arial", 9, "bold"), width=10)
    lbl_text_title.pack(side=tk.LEFT)
    lbl_text = ttk.Label(f_text, text="N/A", wraplength=250, font=("Arial", 8))
    lbl_text.pack(side=tk.LEFT, padx=5)
    lbl_text_w = ttk.Label(f_text, text="(W: 0%)", foreground="gray", font=("Arial", 8))
    lbl_text_w.pack(side=tk.LEFT)

    ttk.Separator(analysis_f, orient='horizontal').pack(fill='x', pady=5)

    final_section = ttk.Frame(analysis_f)
    final_section.pack(fill=tk.X, pady=2)

    lbl_final = ttk.Label(final_section, text="Final Decision: N/A (0%)", font=("Arial", 10, "bold"))
    lbl_final.pack(anchor=tk.W, pady=1)

    scores_frame = ttk.Frame(final_section)
    scores_frame.pack(fill=tk.X, pady=2)
    lbl_scores_title = ttk.Label(scores_frame, text="Emotion Scores:", font=("Arial", 8, "bold"))
    lbl_scores_title.pack(side=tk.LEFT)
    lbl_scores = ttk.Label(scores_frame, text="", font=("Arial", 8))
    lbl_scores.pack(side=tk.LEFT, padx=5)

    lbl_spam = ttk.Label(final_section, text="IDENTIFYING SPAM: 0%", font=("Arial", 9, "bold"), foreground="black")
    lbl_spam.pack(anchor=tk.W, pady=(2, 1))

    lbl_process_time = ttk.Label(final_section, text="", font=("Arial", 8), foreground="gray")
    lbl_process_time.pack(anchor=tk.W)

    weights_f = ttk.LabelFrame(right_panel, text="Analysis Weights", padding=5)
    weights_f.pack(fill=tk.X, pady=3)
    weight_labels = {}

    def on_slider_release(_=None):
        total = sum(v.get() for v in config.emotion_weight_vars.values()) or 1
        for k, v in config.emotion_weight_vars.items(): v.set(v.get() / total * 100); weight_labels[k].config(
            text=f"{v.get():.0f}%")
        save_settings_handler()

    for key, text in {'cnn': 'Model', 'logic': 'Logic', 'text': 'STT'}.items():
        f = ttk.Frame(weights_f);
        f.pack(fill=tk.X, pady=1);
        ttk.Label(f, text=text, width=6).pack(side=tk.LEFT)
        slider = tk.Scale(f, from_=0, to=100, orient=tk.HORIZONTAL, variable=config.emotion_weight_vars[key],
                          showvalue=0, length=150)
        slider.pack(side=tk.LEFT, expand=True, fill=tk.X);
        slider.bind("<ButtonRelease-1>", on_slider_release)
        lbl = ttk.Label(f, width=5);
        lbl.pack(side=tk.RIGHT, padx=5);
        weight_labels[key] = lbl

    src_f = ttk.LabelFrame(right_panel, text="Audio Source", padding=5)
    src_f.pack(fill=tk.X, pady=3)
    int_f = ttk.Frame(src_f)
    int_f.pack(fill=tk.X)
    int_cb = ttk.Combobox(int_f, textvariable=config.internal_audio_device_var, state=tk.DISABLED, width=30)

    def on_src_change():
        is_internal = config.audio_source_var.get() == config.AUDIO_SOURCE_INTERNAL
        int_cb.config(state=tk.NORMAL if is_internal else tk.DISABLED)
        if is_internal: populate_internal_audio_devices(int_cb, sd)
        update_audio_source(sd)
        save_settings_handler()

    ttk.Radiobutton(src_f, text=config.AUDIO_SOURCE_EXTERNAL, variable=config.audio_source_var,
                    value=config.AUDIO_SOURCE_EXTERNAL, command=on_src_change).pack(anchor=tk.W)
    ttk.Radiobutton(int_f, text=config.AUDIO_SOURCE_INTERNAL, variable=config.audio_source_var,
                    value=config.AUDIO_SOURCE_INTERNAL, command=on_src_change).pack(side=tk.LEFT)
    int_cb.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    int_cb.bind("<<ComboboxSelected>>", lambda e: on_src_change())

    leg_f = ttk.LabelFrame(right_panel, text="Color Legend", padding=5);
    leg_f.pack(fill=tk.X, pady=3)
    colors_frame = ttk.Frame(leg_f)
    colors_frame.pack(fill=tk.X)
    for i, (name, color) in enumerate(config.EMO_COL.items()):
        if name not in ['__processing__', 'silent']:
            f = ttk.Frame(colors_frame)
            f.pack(side=tk.LEFT, padx=5)
            ttk.Label(f, background=color, width=2).pack(side=tk.LEFT, padx=2)
            ttk.Label(f, text=name.capitalize(), font=("Arial", 8)).pack(side=tk.LEFT)

    log_f = ttk.LabelFrame(right_panel, text="Transcription Log", padding=5)
    log_f.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=3)
    sb = ttk.Scrollbar(log_f, command=log_text.yview)
    log_text.config(yscrollcommand=sb.set)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, squeeze=False)
    fig.set_constrained_layout(True)
    axes[0, 0].set_title("Speaker 0")
    axes[1, 0].set_title("Speaker 1")
    canvas_f = ttk.Frame(speakers_frame)
    canvas_f.pack(fill=tk.BOTH, expand=True)
    canvas = FigureCanvasTkAgg(fig, master=canvas_f)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    plotted_info = {}

    def on_pick(event):
        if event.artist in plotted_info: show_segment_details(plotted_info[event.artist], root_window)

    fig.canvas.mpl_connect('pick_event', on_pick)

    last_emotion_frame = ttk.LabelFrame(speakers_frame, text="Last Detected Emotions", padding=5)
    last_emotion_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
    last_emotion_labels = []
    for i in range(config.MAX_GUI_SPK):
        frame = ttk.Frame(last_emotion_frame);
        frame.pack(side=tk.LEFT, padx=10, expand=True);
        ttk.Label(frame, text=f"Speaker {i}:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        emo_label = ttk.Label(frame, text="N/A", font=("Arial", 10), relief="solid", padding=5, width=10)
        emo_label.pack(side=tk.LEFT)
        last_emotion_labels.append(emo_label)

    speaker_info_container = ttk.Frame(speakers_frame, padding=(0, 5, 0, 0))
    speaker_info_container.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
    all_cur = []
    for i in range(config.MAX_GUI_SPK):
        f = ttk.LabelFrame(speaker_info_container, text=f"Speaker {i} - Current", padding=5)
        f.grid(row=0, column=i, padx=3, sticky="ew")
        speaker_info_container.grid_columnconfigure(i, weight=1)
        cl = ttk.Label(f, text="Status: ...", font=("Arial", 9))
        cl.pack(anchor=tk.W, fill=tk.X)
        all_cur.append(cl)

    threading.Thread(target=audio_loop, args=(sr.Recognizer(), log_text), daemon=True).start()

    auto_scroll = tk.BooleanVar(value=True)
    scroll_offset = tk.DoubleVar(value=0)
    slider = add_scroll_slider(fig, lambda val: (
        scroll_offset.set(float(val)), auto_scroll.set(slider.valmax <= 0 or val >= slider.valmax - 0.2)), Slider)

    root_window.gui_vars = {'lbl_cnn': lbl_cnn, 'lbl_cnn_w': lbl_cnn_w, 'lbl_logic': lbl_logic,
                            'lbl_logic_w': lbl_logic_w, 'lbl_text': lbl_text, 'lbl_text_w': lbl_text_w,
                            'lbl_final': lbl_final, 'lbl_spam': lbl_spam, 'lbl_scores': lbl_scores,
                            'lbl_process_time': lbl_process_time,
                            'lbl_model_overall': lbl_model_overall,
                            'lbl_model_overall_weight': lbl_model_overall_weight,
                            'lbl_logic_overall': lbl_logic_overall,
                            'lbl_logic_overall_weight': lbl_logic_overall_weight,
                            'lbl_stt_overall': lbl_stt_overall, 'lbl_stt_overall_weight': lbl_stt_overall_weight}

    update_interval = getattr(config, 'GUI_UPDATE_INTERVAL', 250)
    root_window.anim = FuncAnimation(fig,
                                     lambda f: refresh_gui(win=root_window, fig=fig, axes=axes, cur_lbls=all_cur,
                                                           last_emo_lbls=last_emotion_labels, slider=slider,
                                                           auto_scroll=auto_scroll, offset=scroll_offset,
                                                           plotted=plotted_info, np=np, mcolors=mcolors),
                                     interval=update_interval,
                                     cache_frame_data=False)

    root_window.after(100, on_src_change)
    root_window.after(200, on_slider_release)
    root_window.save_settings = save_settings_handler
    root_window.generate_report = generate_report_handler


def refresh_gui(win, fig, axes, cur_lbls, last_emo_lbls, slider, auto_scroll, offset, plotted, np, mcolors):
    if not win.winfo_exists(): return

    if not hasattr(refresh_gui, 'call_count'):
        refresh_gui.call_count = 0
    refresh_gui.call_count += 1

    now = time.time()

    if hasattr(win, 'distress_status_label') and hasattr(config, 'current_distress_status'):
        for speaker_id, status in config.current_distress_status.items():
            if status['at_risk']:
                win.distress_status_label.config(
                    text=f"At Risk: YES - Speaker {speaker_id}",
                    foreground="black"
                )
                duration_min = status['duration'] / 60
                win.distress_detail_label.config(
                    text=f"{status['emotion'].upper()} for {duration_min:.1f} minutes ({status['confidence']:.0%} conf)",
                    foreground="black"
                )
                break
        else:
            win.distress_status_label.config(text="At Risk: No", foreground="black")
            win.distress_detail_label.config(text="Status: All speakers OK", foreground="black")

    for i in range(config.MAX_GUI_SPK):
        if i < len(cur_lbls):
            cur_lbls[i].config(text=f"Status: {config._current_1s_emotion_cache_spk.get(i, {'text': '...'})['text']}")
        if i < len(last_emo_lbls):
            last_info = config._last_significant_emotion_cache_spk.get(i, {'emotion': 'N/A', 'time': 0})
            if now - last_info['time'] < config.RESET_TO_NEUTRAL_AFTER:
                emotion = last_info['emotion']
                confidence = last_info.get('confidence', 0)
                display_text = f"{emotion.upper()} ({confidence * 100:.0f}%)" if emotion != 'N/A' and confidence > 0 else emotion.upper()
                base_color = config.EMO_COL.get(emotion, "lightgray")
                last_emo_lbls[i].config(text=display_text, background=base_color)
            else:
                last_emo_lbls[i].config(text="N/A", background="lightgray")

    total_dur = sum(len(s["audio"]) for s in config.segments_data) / config.PEAK_SR
    new_max = max(0, total_dur - config.SCROLL_WINDOW_SEC)
    if abs(new_max - slider.valmax) > 0.1:
        slider.valmax = new_max
        slider.ax.set_xlim(0, new_max or 1)
    if auto_scroll.get() and abs(slider.val - new_max) > 0.2:
        slider.set_val(new_max)

    start_s, end_s = int(offset.get() * config.PEAK_SR), int((offset.get() + config.SCROLL_WINDOW_SEC) * config.PEAK_SR)

    for ax in axes.flatten():
        ax.cla()
        ax.set_facecolor("#E0E0E0")
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.set_xlim(start_s, end_s)

    axes[0, 0].set_title("Speaker 0")
    axes[1, 0].set_title("Speaker 1")
    plotted.clear()

    Y_POSITIONS = [0.8, 0.6, 0.4, 0.2, -0.2, -0.4, -0.6, -0.8]
    cur_s = 0

    segments_to_display = list(config.segments_data)
    if hasattr(config, 'MAX_SEGMENTS_TO_DISPLAY') and len(segments_to_display) > config.MAX_SEGMENTS_TO_DISPLAY:
        segments_to_display = segments_to_display[-config.MAX_SEGMENTS_TO_DISPLAY:]

    for seg in segments_to_display:
        seg_len = len(seg["audio"])
        s_map = config.segment_id_map.get(seg['id'], {})
        sid = s_map.get('sid', -1)

        if cur_s + seg_len > start_s and cur_s < end_s and 0 <= sid < config.MAX_GUI_SPK:
            emo, conf = s_map.get('emotion_state', '__processing__'), s_map.get('emotion_confidence', 0.5)
            alpha = 0.7
            if emo not in ['__processing__', 'silent']:
                alpha = max(0.2, min(1.0, conf * 1.5))

            color = mcolors.to_rgba(config.EMO_COL.get(emo, "#9370DB"), alpha=alpha)
            line, = axes[sid, 0].plot(np.arange(cur_s, cur_s + seg_len), seg["audio"], color=color, lw=0.8, picker=5)
            plotted[line] = seg["id"]

            if words := s_map.get("words", []):
                s_per_w = seg_len / len(words)
                for j, w in enumerate(words):
                    display_word = w[:15]
                    if any('\u0590' <= c <= '\u05FF' for c in display_word):
                        display_word = display_word[::-1]
                    axes[sid, 0].text(cur_s + s_per_w * (j + 0.5), Y_POSITIONS[j % len(Y_POSITIONS)], display_word,
                                      ha='center', va='center', fontsize=8, clip_on=True, alpha=0.8)
        cur_s += seg_len

    last_id = None
    all_segment_ids = list(config.segment_id_map.keys())

    if all_segment_ids:
        for sid in reversed(all_segment_ids[-50:]):
            if sid in config.segment_id_map:
                seg_data = config.segment_id_map[sid]
                if 'emotion_details' in seg_data and seg_data.get('emotion_state') not in ['__processing__', 'silent']:
                    details = seg_data['emotion_details']
                    if 'final_decision' in details:
                        last_id = sid
                        break

    if last_id and last_id in config.segment_id_map and 'emotion_details' in config.segment_id_map[last_id]:
        details = config.segment_id_map[last_id]['emotion_details']
        gui_vars = win.gui_vars

        cnn_analysis = details.get('cnn_analysis', {})
        logic_analysis = details.get('logic_analysis', {})
        text_analysis = details.get('text_analysis', {})

        emo = details.get('final_decision', 'neutral')
        conf = details.get('final_confidence', 0)
        fw = details.get('final_weights', {'cnn': 0.33, 'logic': 0.33, 'text': 0.34})

        cnn_reason = cnn_analysis.get('reason', 'N/A')
        gui_vars['lbl_cnn'].config(text=f"{cnn_reason}")
        gui_vars['lbl_cnn_w'].config(text=f"(W: {fw.get('cnn', 0) * 100:.0f}%)")

        logic_reason = logic_analysis.get('reason', 'N/A')
        gui_vars['lbl_logic'].config(text=f"{logic_reason}")
        gui_vars['lbl_logic_w'].config(text=f"(W: {fw.get('logic', 0) * 100:.0f}%)")

        text_reason = text_analysis.get('reason', 'N/A')
        gui_vars['lbl_text'].config(text=f"{text_reason}")
        gui_vars['lbl_text_w'].config(text=f"(W: {fw.get('text', 0) * 100:.0f}%)")

        gui_vars['lbl_final'].config(text=f"Final Decision: {emo.capitalize()} ({conf * 100:.0f}%)")

        if 'lbl_model_overall' in gui_vars:
            cnn_dist = cnn_analysis.get('dist', {})
            if cnn_dist:
                cnn_top_emo = max(cnn_dist.items(), key=lambda x: x[1])[0]
                gui_vars['lbl_model_overall'].config(text=cnn_top_emo.capitalize())
                gui_vars['lbl_model_overall_weight'].config(text=f"(Weight {fw.get('cnn', 0) * 100:.0f}%)")

            logic_dist = logic_analysis.get('dist', {})
            if logic_dist:
                logic_top_emo = max(logic_dist.items(), key=lambda x: x[1])[0]
                gui_vars['lbl_logic_overall'].config(text=logic_top_emo.capitalize())
                gui_vars['lbl_logic_overall_weight'].config(text=f"(Weight {fw.get('logic', 0) * 100:.0f}%)")

            text_dist = text_analysis.get('dist', {})
            if text_dist:
                text_top_emo = max(text_dist.items(), key=lambda x: x[1])[0]
                gui_vars['lbl_stt_overall'].config(
                    text=f"{text_top_emo.capitalize()} ({text_dist[text_top_emo] * 100:.0f}%)")
                gui_vars['lbl_stt_overall_weight'].config(text=f"(Weight {fw.get('text', 0) * 100:.0f}%)")

        if 'lbl_scores' in gui_vars and gui_vars['lbl_scores']:
            final_scores = details.get('final_scores', {})
            scores_text = " | ".join([f"{k.capitalize()}: {v * 100:.0f}%" for k, v in
                                      sorted(final_scores.items(), key=lambda x: x[1], reverse=True)])
            gui_vars['lbl_scores'].config(text=scores_text)

        spam_info = details.get('spam_detection', {})
        spam_confidence = spam_info.get('confidence', 0)
        spam_text = f"IDENTIFYING SPAM: {spam_confidence * 100:.0f}%"
        spam_color = "red" if spam_confidence > 0.5 else "black"

        if 'lbl_spam' in gui_vars and gui_vars['lbl_spam']:
            gui_vars['lbl_spam'].config(text=spam_text, foreground=spam_color)

        if 'lbl_process_time' in gui_vars and gui_vars['lbl_process_time']:
            gui_vars['lbl_process_time'].config(text="")

    else:
        if hasattr(win, 'gui_vars'):
            gui_vars = win.gui_vars
            gui_vars['lbl_cnn'].config(text="N/A")
            gui_vars['lbl_cnn_w'].config(text="(W: 0%)")
            gui_vars['lbl_logic'].config(text="N/A")
            gui_vars['lbl_logic_w'].config(text="(W: 0%)")
            gui_vars['lbl_text'].config(text="N/A")
            gui_vars['lbl_text_w'].config(text="(W: 0%)")
            gui_vars['lbl_final'].config(text="Final Decision: N/A (0%)")
            if 'lbl_spam' in gui_vars and gui_vars['lbl_spam']:
                gui_vars['lbl_spam'].config(text="IDENTIFYING SPAM: 0%", foreground="black")
            if 'lbl_scores' in gui_vars and gui_vars['lbl_scores']:
                gui_vars['lbl_scores'].config(text="")
            if 'lbl_process_time' in gui_vars and gui_vars['lbl_process_time']:
                gui_vars['lbl_process_time'].config(text="")
            if 'lbl_model_overall' in gui_vars:
                gui_vars['lbl_model_overall'].config(text="N/A")
                gui_vars['lbl_model_overall_weight'].config(text="")
                gui_vars['lbl_logic_overall'].config(text="N/A")
                gui_vars['lbl_logic_overall_weight'].config(text="")
                gui_vars['lbl_stt_overall'].config(text="N/A")
                gui_vars['lbl_stt_overall_weight'].config(text="")

    try:
        fig.canvas.draw_idle()
    except Exception:
        pass