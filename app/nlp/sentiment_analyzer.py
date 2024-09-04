from io import StringIO
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import openai
from openai import OpenAI
import pandas as pd

from app.config import settings


openai_client = OpenAI(api_key=settings.openai_api_key)

NLTK_STOP_WORDS = set(stopwords.words('english'))
analyzer = SentimentIntensityAnalyzer()


def init_nltk() -> None:
    nltk.download('all')


def generate_polarity_sentiment(text: str, pre_process_text: bool = True) -> dict:
    init_nltk()
    if pre_process_text:
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in NLTK_STOP_WORDS]
        text = ' '.join(filtered_tokens)
    polarity_metrics = analyzer.polarity_scores(text)
    return polarity_metrics


def generate_emotional_sentiment(text: str) -> tuple[pd.DataFrame, dict]:
    prompt = f"""
    Hi ChatGPT! I'd like your help quantifying the range and degree of emotions expressed in a snippet of text. In the text snippet below, please assess the degree of emotion in 10 dimensions: (1) Joy, (2) Love, (3) Empathy, (4) Curiosity, (5) Sadness, (6) Anger, (7) Fear, (8) Disgust, (9) Surprise, (10) Confusion. 
    
    For each of these 10 emotions, please return a decimal value between 0 and 1 indicating the 'score' (or 'weight' / 'intensity') of that emotion. The 'score' should be precise to 2 decimal places.
    
    Note: there need not be any relationship between the individual emotional metric scores. In aggregate, they may all add up to less than 1.0, or their combined total may greatly exceed 1.0 - each emotional metric is independent of the others. 
    
    Provide the response in CSV format with 3 columns for (1) 'emotion', (2) 'score', (3) 'explanation' for score. If column (3) contains any quotes or apostrophes, please be sure to escape the text properly. 

    Here is the snippet of text:

    {text}
    """

    prompt_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )

    success = False
    if prompt_response.choices:
        # print('stage 1')
        if len(prompt_response.choices) > 0:
            # print('stage 2')
            if prompt_response.choices[0].message:
                # print('stage 3')
                if prompt_response.choices[0].message.content:
                    # print('stage 4')
                    response_content = prompt_response.choices[0].message.content
                    response_content_bits = response_content.split('```')
                    if len(response_content_bits) == 3:
                        # print(f'attempting to assing csv_string to response_content_bits[1]={response_content_bits[1]}')
                        csv_string = response_content_bits[1]
                        if csv_string.startswith('csv\n'):
                            csv_string = csv_string.removeprefix('csv\n')
                        try:
                            # print(f'attempting to call StringIO on csv_string')
                            csv_string_io = StringIO(csv_string)
                            # print(f'attempting to create df out of csv_string_io')
                            emo_df = pd.read_csv(csv_string_io, sep=",")
                            print(f'successful emo_df={emo_df}')
                            success = True
                        except Exception as e:
                            print(f'failure to init StringIO for csv_string={csv_string}')
                            return None, {}
    
    if not success:
        print(f'problem parsing prompt_response={prompt_response}')    
        return None, {}
    
    if emo_df is None or 'emotion' not in emo_df or 'score' not in emo_df:
        print(f'unable to convert `emo_df` to dict after analyzing text in generate_emotional_sentiment, original text={text}')
        return None, {}
    slim_emo_df = emo_df[['emotion', 'score']]
    emo_dict = dict(zip(slim_emo_df.emotion, slim_emo_df.score))
    
    return emo_df, emo_dict

