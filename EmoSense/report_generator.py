"""
Report generation module for EmoSense.

Generates summary text reports with per-speaker emotion breakdowns,
sample transcripts, and distress detection status.
"""

import os
from datetime import datetime
import config


def generate_summary_report():
    """Generate a session summary report.

    Creates a text file in the reports/ directory containing emotion
    statistics per speaker, sample transcripts, and distress alerts.

    Returns:
        Path to the generated report file, or None if no data available.
    """
    try:
        if not config.segments_data or not config.segment_id_map:
            return None

        if not os.path.exists("reports"):
            os.makedirs("reports")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/emotion_report_{timestamp}.txt"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("EMOSENSE EMOTION ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Emotion Sensitivity: {config.EMOTION_SENSITIVITY}\n\n")

            speaker_emotions = {}
            total_segments = 0

            for seg_id, seg_data in config.segment_id_map.items():
                if 'emotion_state' in seg_data and seg_data['emotion_state'] not in ['__processing__', 'silent']:
                    speaker_id = seg_data.get('sid', -1)
                    if speaker_id not in speaker_emotions:
                        speaker_emotions[speaker_id] = {
                            'happy': 0, 'sad': 0, 'angry': 0, 'neutral': 0,
                            'total': 0, 'transcripts': []
                        }

                    emotion = seg_data['emotion_state']
                    speaker_emotions[speaker_id][emotion] = speaker_emotions[speaker_id].get(emotion, 0) + 1
                    speaker_emotions[speaker_id]['total'] += 1
                    total_segments += 1

                    if 'words' in seg_data and seg_data['words']:
                        text = ' '.join(seg_data['words'])
                        speaker_emotions[speaker_id]['transcripts'].append({
                            'text': text,
                            'emotion': emotion,
                            'confidence': seg_data.get('emotion_confidence', 0)
                        })

            f.write(f"Total analyzed segments: {total_segments}\n")
            f.write(f"Number of speakers: {len(speaker_emotions)}\n\n")

            for speaker_id, data in speaker_emotions.items():
                f.write(f"\n--- Speaker {speaker_id} ---\n")
                f.write(f"Total segments: {data['total']}\n")
                f.write("Emotion breakdown:\n")

                for emotion in ['happy', 'sad', 'angry', 'neutral']:
                    count = data[emotion]
                    percentage = (count / data['total'] * 100) if data['total'] > 0 else 0
                    f.write(f"  {emotion.capitalize()}: {count} ({percentage:.1f}%)\n")

                f.write("\nSample transcripts:\n")
                for i, transcript in enumerate(data['transcripts'][:5]):
                    f.write(f"  [{transcript['emotion'].upper()}] {transcript['text']}\n")

                f.write("\n")

            f.write("\nDistress Detection\n")
            any_distress = False
            for speaker_id, status in config.current_distress_status.items():
                if status['at_risk']:
                    any_distress = True
                    f.write(f"Speaker {speaker_id}: AT RISK - {status['emotion']} "
                            f"for {status['duration'] / 60:.1f} minutes\n")

            if not any_distress:
                f.write("No speakers currently at risk.\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("End of Report\n")

        return filename

    except Exception as e:
        print(f"Error generating report: {e}")
        return None