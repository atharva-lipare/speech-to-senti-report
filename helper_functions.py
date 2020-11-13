import subprocess
import os
import pandas as pd
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import SpeechToTextV1
from ibm_watson import PersonalityInsightsV3
from ibm_watson import ToneAnalyzerV3
from reportlab.lib.units import cm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.stem import WordNetLemmatizer
from tinytag import TinyTag
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from datetime import datetime as dt
from reportlab.lib.pagesizes import A4
from reportlab.lib import utils
from wordcloud import WordCloud
from deepface import DeepFace

plt.style.use('ggplot')

VIDEO_PATH = ''
REPORT_PATH = ''
AUDIO_PATH = os.path.abspath(os.path.join('.', 'temp', 'audio.flac'))
SPLIT_PATH = os.path.abspath(os.path.join('.', 'temp', 'splits'))
FIGURES_PATH = os.path.abspath(os.path.join('.', 'temp', 'figures'))
split_time = 0
all_text = ''
transcript_list = []
tones = []
emotions = {}
is_personality_successful = False

IBM_SPEECH_TO_TEXT_API_KEY = '{apikey}'
IBM_SPEECH_TO_TEXT_SERVICE_URL = '{url}'
IBM_PERSONALITY_INSIGHTS_API_KEY = '{apikey}'
IBM_PERSONALITY_INSIGHTS_SERVICE_URL = '{url}'
IBM_TONE_ANALYZER_API_KEY = '{apikey}'
IBM_TONE_ANALYZER_SERVICE_URL = '{url}'


def get_audio():
    """
    Gets audio.flac from video.mp4
    """
    global VIDEO_PATH
    # ffmpeg -i "/content/drive/My Drive/colab_drive/speech-sentiment/video/ElevatorPitchWinner.mp4" -f flac
    # -sample_fmt s16 -ar 16000 audio-file.flac

    command = ['mkdir', os.path.abspath(os.path.join('.', 'temp'))]
    print(*command)
    subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)

    command = ['ffmpeg', '-i', VIDEO_PATH, '-f', 'flac', '-sample_fmt', 's16', '-ar', '16000', AUDIO_PATH, '-y']
    print(*command)
    subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
    print('getAudio finished')


def split_audio():
    """
    splits audio.flac into 4 equal parts
    """
    global split_time
    command = ['mkdir', SPLIT_PATH]
    print(*command)
    subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)

    command = ['mkdir', FIGURES_PATH]
    print(*command)
    subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)

    tag = TinyTag.get(AUDIO_PATH)
    split_time = int(tag.duration / 4) + 1

    # ffmpeg -i ./audio-file.flac -f segment -segment_time 30 -c copy ./splits/out%03d.flac
    command = ['ffmpeg', '-i', AUDIO_PATH, '-f', 'segment', '-segment_time', str(split_time), '-c', 'copy',
               os.path.abspath(os.path.join(SPLIT_PATH, 'out%03d.flac')), '-y']
    print(*command)
    subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
    print('split_audio finished')


def get_transcript_list():
    """
    gets transcript using IBM Speech To Text
    https://www.ibm.com/cloud/watson-speech-to-text
    """
    global all_text, transcript_list
    authenticator = IAMAuthenticator(IBM_SPEECH_TO_TEXT_API_KEY)
    speech_to_text = SpeechToTextV1(
        authenticator=authenticator
    )
    speech_to_text.set_service_url(IBM_SPEECH_TO_TEXT_SERVICE_URL)

    for entry in sorted(os.listdir(SPLIT_PATH)):
        with open(os.path.join(SPLIT_PATH, entry), 'rb') as audio_file:
            result_text = speech_to_text.recognize(
                audio=audio_file,
                content_type='audio/flac',
                model='en-US_BroadbandModel',
                smart_formatting=True
            )
        transcript = ""
        for x in result_text.get_result()['results']:
            transcript += x['alternatives'][0]['transcript']
        transcript_list.append(transcript)
        all_text += transcript
    print('get_transcript_list finished')


def get_vader_sentiment():
    """
    gets sentiment scores of audio segment using VADER and saves as sentiment.png
    https://github.com/cjhutto/vaderSentiment
    """
    global transcript_list
    sia = SentimentIntensityAnalyzer()
    sentiment_scores_list = []
    for text in transcript_list:
        sentiment_scores_list.append(sia.polarity_scores(text))
    y_axis = {'pos': [x['pos'] for x in sentiment_scores_list], 'neg': [x['neg'] for x in sentiment_scores_list],
              'compound': [x['compound'] for x in sentiment_scores_list]}
    x_axis = range(1, 1 + len(y_axis['neg']))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(x_axis, y_axis['pos'], label='positive sentiment')
    ax.plot(x_axis, y_axis['neg'], label='negative sentiment')
    ax.plot(x_axis, y_axis['compound'], label='compound sentiment')
    ax.set_xticks(range(1, 1 + len(y_axis['neg'])))
    ax.legend()
    plt.title("Sentiment versus Time")
    fig.savefig(os.path.join('.', 'temp', 'figures', 'sentiment.png'), bbox_inches='tight')
    print('get_vader_sentiment finished')


def get_word_cloud():
    """
    generates word cloud of transcript and saves as cloud.png
    https://github.com/amueller/word_cloud
    """
    global all_text
    wordcloud = WordCloud(background_color="white").generate(all_text)
    wordcloud.to_file(os.path.join(FIGURES_PATH, 'cloud.png'))


def get_word_frequency_v1():
    """
    generates horizontal bar graph of frequency of top 10 used words and saves as frequency.png
    """
    global all_text
    lemmatizer = WordNetLemmatizer()
    word_list = nltk.word_tokenize(all_text)
    lema_all_text = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    all_words = nltk.tokenize.word_tokenize(lema_all_text)
    stopwords = nltk.corpus.stopwords.words('english')
    all_word_except_stop_dist = nltk.FreqDist(w.lower() for w in all_words if w.lower() not in stopwords)
    most_common = all_word_except_stop_dist.most_common(10)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.barh(list(dict(most_common).keys()), list(dict(most_common).values()))
    plt.title("Most frequently used words")
    fig.savefig(os.path.join('.', 'temp', 'figures', 'frequency.png'), bbox_inches='tight')


def get_personality_insights():
    """
    generates horizontal bar graph of personality scores using IBM Personality Insights
    https://www.ibm.com/watson/services/personality-insights/
    """
    global all_text, is_personality_successful
    authenticator = IAMAuthenticator(IBM_PERSONALITY_INSIGHTS_API_KEY)
    personality_insights = PersonalityInsightsV3(
        version='2017-10-13',
        authenticator=authenticator
    )
    personality_insights.set_service_url(IBM_PERSONALITY_INSIGHTS_SERVICE_URL)
    try:
        profile = personality_insights.profile(all_text, accept='application/json', raw_scores=True).get_result()
    except:
        is_personality_successful = False
        return

    is_personality_successful = True
    attributes = ['personality', 'needs', 'values']
    for attr in attributes:
        list_val = []
        x_axis = []
        for quality in profile[attr]:
            x_axis.append(quality['name'])
            list_val.append([quality['percentile'] * 100, quality['raw_score'] * 100])
        pd.DataFrame(list_val, columns=['Percentile', 'Raw Score'], index=x_axis) \
            .plot(kind='barh', figsize=(8, 6), title='Scores for ' + attr.capitalize()) \
            .get_figure().savefig(os.path.join('.', 'temp', 'figures', attr + '.png'), bbox_inches='tight')

    print('get_personality_insights finished')


def get_tone_analysis():
    """
    gets tones/emotions used in transcript using IBM Watson Tone Analyser
    https://www.ibm.com/watson/services/tone-analyzer/
    """
    global split_time, transcript_list, tones
    authenticator = IAMAuthenticator(IBM_TONE_ANALYZER_API_KEY)
    tone_analyzer = ToneAnalyzerV3(
        version='2017-09-21',
        authenticator=authenticator
    )
    tone_analyzer.set_service_url(IBM_TONE_ANALYZER_SERVICE_URL)

    tone_list = []
    for x in transcript_list:
        tone_list.append(tone_analyzer.tone({'text': x}, content_type='application/json').get_result())
    time_list = []
    tone_name_list = []
    i = 1
    tones = []
    for tone in tone_list:
        if len(tone['document_tone']['tones']) > 0:
            for toneType in tone['document_tone']['tones']:
                time_list.append([split_time * (i - 1), split_time * i])
                tone_name_list.append(toneType['tone_name'])
                temp_str = 'Tone: {} was detected from: {} sec to {} sec'. \
                    format(toneType['tone_name'], split_time * (i - 1), split_time * i)
                tones.append(temp_str)
        i += 1
    print('get_tone_analysis finished')


def get_emotions():
    """
    gets the emotions recognised from the face of the subject. Uses deepface library.
    I extract 2 frames per second from the video and then get the proportions of emotions used.
    https://github.com/serengil/deepface
    """
    global VIDEO_PATH, emotions
    command = ['mkdir', os.path.abspath(os.path.join('.', 'temp', 'frames'))]
    print(*command)
    subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)

    #  ffmpeg -y -i .\latest_test_vid.mov -r 2 frames\image.%06d.png
    command = ['ffmpeg', '-y', '-i', VIDEO_PATH, '-r', '2', os.path.abspath(os.path.join('.', 'temp', 'frames',
                                                                                         'frame.%06d.png'))]
    print(*command)
    subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)

    frames = [os.path.join('.', 'temp', 'frames', x.name) for x in os.scandir(os.path.join('.', 'temp', 'frames'))]
    obj = DeepFace.analyze(frames, actions=['emotion'], enforce_detection=False)

    emotions = {}
    for x in obj:
        try:
            emotions[obj[x]['dominant_emotion']] += 1
        except:
            emotions[obj[x]['dominant_emotion']] = 1
    total_frames = len(obj)
    for x in emotions:
        emotions[x] = int(emotions[x] / total_frames * 1000) / 10

    labels = list(emotions.keys())
    sizes = list(emotions.values())
    explode = [0.1] * len(emotions)
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    fig1.savefig(os.path.join('.', 'temp', 'figures', 'emotions.png'), bbox_inches='tight')

    command = ['del', os.path.abspath(os.path.join('.', 'temp', 'frames', '*.png'))]
    print(*command)
    subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)


def get_report():
    """
    generates PDF report and saves file in user defined out directory using reportlab
    https://pypi.org/project/reportlab/
    """
    global REPORT_PATH, tones, emotions
    pdf = canvas.Canvas(os.path.join(REPORT_PATH, 'report_' + dt.now().strftime("%Y_%m_%d_%H_%M_%S") + '.pdf'))
    pdf.pagewidth = A4[0]  # 595.2755905511812
    pdf.pageheight = A4[1]  # 841.8897637795277
    pdf.setTitle('Report')
    pdfmetrics.registerFont(TTFont('arial', 'Arial.ttf'))
    pdfmetrics.registerFont(TTFont('calibri', 'Calibri.ttf'))
    pdf.setFont('arial', 32)
    pdf.drawString(60, A4[1] - 75, 'Sentiment Analysis Report')
    pdf.setLineWidth(4)
    pdf.setStrokeColorRGB(83 / 255, 137 / 255, 237 / 255, 1)
    pdf.line(55, A4[1] - 90, 595 - 55, A4[1] - 90)
    pdf.setFont('calibri', 18)
    pdf.drawString(60, A4[1] - 160, 'Sentiment Score vs. Time:')
    w, h = utils.ImageReader(os.path.join('.', 'temp', 'figures', 'sentiment.png')).getSize()
    pdf.drawImage(os.path.join('.', 'temp', 'figures', 'sentiment.png'),
                  x=60, y=A4[1] - 530, width=15 * cm, height=h / w * 15 * cm)
    pdf.drawString(60, 290, 'Word Cloud:')
    w, h = utils.ImageReader(os.path.join('.', 'temp', 'figures', 'cloud.png')).getSize()
    pdf.drawImage(os.path.join('.', 'temp', 'figures', 'cloud.png'),
                  x=60, y=50, width=15 * cm, height=h / w * 15 * cm)
    pdf.showPage()

    pdf.setFont('calibri', 18)
    pdf.drawString(60, A4[1] - 100, 'Frequency of words used:')
    w, h = utils.ImageReader(os.path.join('.', 'temp', 'figures', 'frequency.png')).getSize()
    pdf.drawImage(os.path.join('.', 'temp', 'figures', 'frequency.png'),
                  x=60, y=A4[1] - 470, width=15 * cm, height=h / w * 15 * cm)

    if is_personality_successful:
        pdf.drawString(60, A4[1] - 530, 'Personality Insights:')
        w, h = utils.ImageReader(os.path.join('.', 'temp', 'figures', 'personality.png')).getSize()
        pdf.drawImage(os.path.join('.', 'temp', 'figures', 'personality.png'),
                      x=60, y=A4[1] - 830, width=15 * cm, height=h / w * 15 * cm)
        pdf.showPage()
        pdf.setFont('calibri', 18)
        pdf.drawString(60, A4[1] - 100, 'Personality Insights:')
        w, h = utils.ImageReader(os.path.join('.', 'temp', 'figures', 'needs.png')).getSize()
        pdf.drawImage(os.path.join('.', 'temp', 'figures', 'needs.png'),
                      x=60, y=A4[1] - 430, width=15 * cm, height=h / w * 15 * cm)
        w, h = utils.ImageReader(os.path.join('.', 'temp', 'figures', 'values.png')).getSize()
        pdf.drawImage(os.path.join('.', 'temp', 'figures', 'values.png'),
                      x=60, y=A4[1] - 800, width=15 * cm, height=h / w * 15 * cm)
        pdf.showPage()
    else:
        pdf.showPage()

    pdf.setFont('calibri', 18)
    pdf.drawString(60, A4[1] - 100, 'Tone Insights:')
    pdf.setFont('calibri', 12)
    for i in range(len(tones)):
        pdf.drawString(60, A4[1] - (130 + 20 * i), tones[i])
    pdf.setFont('calibri', 18)
    pdf.drawString(60, A4[1] - (170 + 20 * len(tones)), 'Facial Expressions Insights:')
    pdf.setFont('calibri', 12)
    pdf.drawString(60, A4[1] - (200 + 20 * len(tones)), 'Following were the distribution of expressions over the '
                                                        'duration of the video')
    i = 0
    for x in emotions:
        pdf.drawString(60, A4[1] - (220 + 20 * (i + len(tones))), '{}: {}%'.format(str(x).capitalize(), emotions[x]))
        i += 1
    w, h = utils.ImageReader(os.path.join('.', 'temp', 'figures', 'emotions.png')).getSize()
    pdf.drawImage(os.path.join('.', 'temp', 'figures', 'emotions.png'),
                  x=60, y=A4[1] - (560 + 20 * (i + len(tones))), width=15 * cm, height=h / w * 15 * cm)
    pdf.save()
    print('get_report finished')


def start_analysis(file_name, report_dir):
    global all_text, transcript_list, split_time, VIDEO_PATH, REPORT_PATH, tones, emotions, is_personality_successful
    VIDEO_PATH = file_name
    REPORT_PATH = report_dir
    all_text = ''
    transcript_list = []
    split_time = 0
    tones = []
    emotions = {}
    is_personality_successful = False
    get_audio()
    split_audio()
    get_transcript_list()
    get_vader_sentiment()
    get_word_frequency_v1()
    get_word_cloud()
    get_personality_insights()
    get_tone_analysis()
    get_emotions()
    get_report()
    print('Analysis finished')
