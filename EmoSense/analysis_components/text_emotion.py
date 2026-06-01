"""Text emotion analysis."""

import config
from utils import contains_predominantly_hebrew


def text_based_emotion_analysis(text_input):
    """Analyze emotion from transcribed text.

    Uses keyword matching, laughter detection, RoBERTa NLP pipeline,
    and Hebrew sentiment analysis (HeBERT) to classify emotions.

    Args:
        text_input: Transcribed speech text, or None.

    Returns:
        Tuple of (probability_distribution, reason_string).
    """
    if not text_input or not text_input.strip():
        return {"neutral": 0.6, "happy": 0.13, "sad": 0.13, "angry": 0.14}, "No text"

    is_hebrew = contains_predominantly_hebrew(text_input)

    max_emotion_score = 0
    dominant_emotion = None
    reason = None

    if is_hebrew and not hasattr(config, 'hebrew_sentiment_pipeline'):
        try:
            from transformers import pipeline
            config.hebrew_sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="avichr/heBERT_sentiment_analysis",
                device=-1
            )
        except Exception:
            config.hebrew_sentiment_pipeline = None

    HAPPY_WORDS = {
        'haha', 'hahaha', 'hahahaha', 'lol', 'lmao', 'rofl', 'hehe', 'hihi', 'heehee', 'teehee',
        ':)', ':-)', ':d', ':D', '^_^', 'xD',
        'happy', 'joy', 'joyful', 'glad', 'pleased', 'delighted', 'cheerful', 'excited', 'thrilled',
        'wonderful', 'amazing', 'fantastic', 'great', 'awesome', 'excellent', 'perfect', 'beautiful',
        'love', 'adore', 'enjoy', 'fun', 'funny', 'hilarious', 'smile', 'smiling', 'laugh', 'laughing',
        'yay', 'hooray', 'woohoo', 'yes', 'yeah', 'yup', 'absolutely', 'definitely', 'sure', 'of course',
        'thanks', 'thank you', 'appreciate', 'grateful', 'blessed', 'fortunate', 'lucky',
        'congratulations', 'congrats', 'celebrate', 'party', 'success', 'win', 'winner', 'victory',
        'good', 'nice', 'cool', 'sweet', 'lovely', 'adorable', 'cute', 'charming', 'pleasant',
        'brilliant', 'superb', 'magnificent', 'splendid', 'marvelous', 'terrific', 'fabulous',
        'positive', 'optimistic', 'hopeful', 'enthusiastic', 'passionate', 'proud', 'satisfied',
        'comfortable', 'relaxed', 'peaceful', 'content', 'fulfilled', 'accomplished',
        'חחח', 'חהחה', 'ההה', 'לול', 'אהבה', 'שמח', 'שמחה', 'כיף', 'מצחיק', 'נהדר', 'מעולה',
        'יופי', 'סבבה', 'אחלה', 'מדהים', 'וואו', 'יאי', 'תודה', 'מקסים', 'חמוד', 'כיף',
        'אהבתי', 'נפלא', 'משגע', 'מושלם', 'נהנה', 'נהנית', 'טוב', 'יפה', 'מצוין'
    }

    ANGRY_WORDS = {
        'angry', 'mad', 'furious', 'rage', 'pissed', 'annoyed', 'irritated', 'frustrated', 'upset',
        'hate', 'damn', 'dammit', 'hell', 'wtf', 'ffs', 'bullshit', 'crap', 'stupid', 'idiot',
        'ridiculous', 'unacceptable', 'outrageous', 'disgusting', 'terrible', 'horrible', 'awful',
        'unbelievable', 'seriously', 'kidding me', 'sick of', 'fed up', 'enough', 'stop',
        'annoying', 'bothering', 'disturbing', 'infuriating', 'aggravating', 'exasperating',
        'hostile', 'bitter', 'resentful', 'offended', 'insulted', 'disrespected',
        'unfair', 'unjust', 'wrong', 'rude', 'mean', 'nasty', 'cruel', 'harsh',
        'pathetic', 'worthless', 'useless', 'incompetent', 'foolish', 'moronic',
        'כועס', 'כעס', 'עצבני', 'מעצבן', 'נמאס', 'די', 'מספיק', 'זעם', 'שנאה', 'מתוסכל',
        'עזוב', 'חרא', 'זבל', 'מטומטם', 'דביל', 'אידיוט', 'בולשיט', 'מניאק', 'משוגע',
        'מרגיז', 'מתסכל', 'נורא', 'איום', 'גרוע', 'מחריד', 'מזעזע'
    }

    SAD_WORDS = {
        'sad', 'unhappy', 'depressed', 'down', 'blue', 'cry', 'crying', 'tears', 'sorry', 'apologize',
        'unfortunately', 'sadly', 'regret', 'miss', 'missed', 'lonely', 'alone', 'hurt', 'pain',
        'disappointed', 'disappointment', 'failed', 'failure', 'lost', 'loss', 'grief', 'mourn',
        'heartbroken', 'devastated', 'hopeless', 'helpless', 'tired', 'exhausted', 'drained',
        'difficult', 'hard', 'tough', 'struggle', 'struggling', 'suffered', 'suffering',
        'melancholy', 'gloomy', 'miserable', 'dejected', 'despondent', 'discouraged',
        'unfortunate', 'tragic', 'terrible', 'awful', 'horrible', 'dreadful',
        'empty', 'hollow', 'numb', 'broken', 'shattered', 'crushed', 'defeated',
        'guilty', 'ashamed', 'embarrassed', 'humiliated', 'rejected', 'abandoned',
        'worried', 'anxious', 'stressed', 'overwhelmed', 'burden', 'heavy',
        'עצוב', 'עצובה', 'בוכה', 'דמעות', 'מצטער', 'מצטערת', 'חבל', 'כאב', 'כואב', 'קשה',
        'בדידות', 'בודד', 'מאוכזב', 'נכשל', 'הפסד', 'פספוס', 'עייף', 'מותש', 'מיואש',
        'דכאון', 'דיכאון', 'שבור', 'הרוס', 'מפחד', 'חושש', 'דואג', 'מודאג', 'לחוץ'
    }

    try:
        text_lower = text_input.lower()

        happy_count = sum(1 for word in HAPPY_WORDS if word in text_lower)
        angry_count = sum(1 for word in ANGRY_WORDS if word in text_lower)
        sad_count = sum(1 for word in SAD_WORDS if word in text_lower)

        laughter_patterns = ['haha', 'hehe', 'hihi', 'hahaha', 'lol', 'lmao', 'rofl', 'חחח', 'ההה']
        has_laughter = any(pattern in text_lower for pattern in laughter_patterns)

        if has_laughter:
            dist = {"happy": 0.7, "angry": 0.1, "sad": 0.1, "neutral": 0.1}
            reason = "Laughter detected"

            if not is_hebrew and config.text_emotion_pipeline:
                try:
                    if len(text_input) > 200:
                        text_input_truncated = text_input[:200]
                    else:
                        text_input_truncated = text_input

                    results = config.text_emotion_pipeline(text_input_truncated)[0]

                    for score in results:
                        label = score['label']
                        confidence = score['score'] * 0.3

                        if label == 'anger' and confidence > 0.15:
                            dist['angry'] += confidence * 0.2
                            dist['happy'] -= confidence * 0.1
                        elif label == 'sadness' and confidence > 0.15:
                            dist['sad'] += confidence * 0.2
                            dist['happy'] -= confidence * 0.1

                except Exception:
                    pass

        elif happy_count > 0 and happy_count > angry_count and happy_count > sad_count:
            dist = {"happy": 0.4, "angry": 0.1, "sad": 0.1, "neutral": 0.4}
            reason = f"Happy words detected ({happy_count})"
        elif angry_count > 0 and angry_count > happy_count and angry_count > sad_count:
            dist = {"happy": 0.1, "angry": 0.4, "sad": 0.1, "neutral": 0.4}
            reason = f"Angry words detected ({angry_count})"
        elif sad_count > 0 and sad_count > happy_count and sad_count > angry_count:
            dist = {"happy": 0.1, "angry": 0.1, "sad": 0.4, "neutral": 0.4}
            reason = f"Sad words detected ({sad_count})"
        else:
            dist = {"happy": 0.15, "angry": 0.15, "sad": 0.15, "neutral": 0.55}
            reason = None

        if is_hebrew and hasattr(config, 'hebrew_sentiment_pipeline') and config.hebrew_sentiment_pipeline:
            try:
                hebrew_results = config.hebrew_sentiment_pipeline(text_input)[0]
                label = hebrew_results['label']
                score = hebrew_results['score']

                if label == 'positive' and score > 0.7:
                    dist['happy'] += 0.3
                    dist['neutral'] -= 0.2
                    dist['sad'] *= 0.5
                    dist['angry'] *= 0.5
                elif label == 'positive':
                    dist['happy'] += 0.15
                    dist['neutral'] -= 0.1
                elif label == 'negative' and score > 0.7:
                    if angry_count > sad_count:
                        dist['angry'] += 0.3
                        dist['sad'] += 0.1
                    else:
                        dist['sad'] += 0.3
                        dist['angry'] += 0.1
                    dist['neutral'] -= 0.3
                    dist['happy'] *= 0.5
                elif label == 'negative':
                    dist['sad'] += 0.15
                    dist['angry'] += 0.1
                    dist['neutral'] -= 0.15

                dominant_emotion = max(dist.items(), key=lambda x: x[1])[0]
                reason = f"Hebrew text: {label} ({score:.2f}), suggests '{dominant_emotion}'"

            except Exception:
                pass

        elif not has_laughter and config.text_emotion_pipeline:
            if len(text_input) > 200:
                text_input = text_input[:200]

            results = config.text_emotion_pipeline(text_input)[0]

            for score in results:
                label = score['label']
                confidence = score['score']

                if confidence < 0.2:
                    continue

                if confidence > max_emotion_score:
                    max_emotion_score = confidence
                    dominant_emotion = label

                emotion_value = confidence * 0.5

                if label == 'joy':
                    dist['happy'] += emotion_value
                    dist['neutral'] -= emotion_value * 0.8

                elif label == 'optimism':
                    dist['happy'] += emotion_value
                    dist['neutral'] -= emotion_value * 0.8

                elif label == 'love':
                    dist['happy'] += emotion_value * 0.9
                    dist['neutral'] -= emotion_value * 0.7

                elif label == 'anger':
                    if confidence > 0.45:
                        dist['angry'] += emotion_value * 0.5
                        dist['neutral'] -= emotion_value * 0.3

                elif label == 'sadness':
                    if confidence > 0.25:
                        dist['sad'] += emotion_value
                        dist['neutral'] -= emotion_value * 0.7

                elif label == 'fear':
                    dist['sad'] += emotion_value * 0.7
                    dist['angry'] += emotion_value * 0.3
                    dist['neutral'] -= emotion_value * 0.8

                elif label == 'surprise':
                    dist['happy'] += emotion_value * 0.4
                    dist['neutral'] += emotion_value * 0.6

                elif label == 'disgust':
                    dist['angry'] += emotion_value * 0.6
                    dist['sad'] += emotion_value * 0.4
                    dist['neutral'] -= emotion_value * 0.8

            if dominant_emotion and max_emotion_score > 0:
                reason = f"Text suggests '{dominant_emotion}' (conf: {max_emotion_score:.2f})"

        for key in dist:
            dist[key] = max(0.05, dist[key])

        total = sum(dist.values())
        final_dist = {k: v / total for k, v in dist.items()}

        for emo in ['happy', 'sad', 'angry']:
            if final_dist[emo] > 0.5:
                excess = final_dist[emo] - 0.5
                final_dist[emo] = 0.5
                final_dist['neutral'] += excess * 0.5

        total = sum(final_dist.values())
        final_dist = {k: v / total for k, v in final_dist.items()}

        final_emotion = max(final_dist.items(), key=lambda x: x[1])[0]
        final_confidence = final_dist[final_emotion]

        if reason is None or (
                not has_laughter and not any(count > 0 for count in [happy_count, angry_count, sad_count])):
            if final_emotion != 'neutral':
                reason = f"Text suggests '{final_emotion}' (conf: {final_confidence:.2f})"
            else:
                reason = "Text unclear"
        elif not has_laughter:
            if "Happy words" in reason and final_emotion != 'happy':
                reason = f"Text suggests '{final_emotion}' (conf: {final_confidence:.2f})"
            elif "Angry words" in reason and final_emotion != 'angry':
                reason = f"Text suggests '{final_emotion}' (conf: {final_confidence:.2f})"
            elif "Sad words" in reason and final_emotion != 'sad':
                reason = f"Text suggests '{final_emotion}' (conf: {final_confidence:.2f})"

        return final_dist, reason

    except Exception as e:
        return {"neutral": 0.6, "happy": 0.13, "sad": 0.13, "angry": 0.14}, f"Text Error: {e}"
