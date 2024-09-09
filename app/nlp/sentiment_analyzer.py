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


def generate_polarity_sentiment(text: str, pre_process_text: bool = True) -> tuple[pd.DataFrame, dict]:
    init_nltk()
    if pre_process_text:
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in NLTK_STOP_WORDS]
        text = ' '.join(filtered_tokens)
    
    polarity_dict = analyzer.polarity_scores(text)

    # kind of a lot of steps to turn the output of polarity_scores into a properly indexed df with coherently named columns, but...
    polarity_df = pd.DataFrame(polarity_dict, index=['score'])
    polarity_df = polarity_df.transpose()
    polarity_df.reset_index(inplace=True)
    polarity_df = polarity_df.rename(columns={'index': 'polarity'})
    # remove 'compound' polarity value, we're never using it
    polarity_df = polarity_df.loc[polarity_df['polarity'] != 'compound']

    return polarity_df, polarity_dict


def generate_emotional_sentiment(text: str) -> tuple[pd.DataFrame, dict]:
    prompt = f"""
    Hi ChatGPT! I'd like your help quantifying the range and intensity of emotions expressed in a snippet of text. The snippet of text is a line spoken by a character in a film or television series.
    
    In the text snippet below, please assess the degree of emotion in 10 dimensions: (1) Joy, (2) Love, (3) Empathy, (4) Curiosity, (5) Sadness, (6) Anger, (7) Fear, (8) Disgust, (9) Surprise, (10) Confusion. 
    
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
        if len(prompt_response.choices) > 0:
            if prompt_response.choices[0].message:
                if prompt_response.choices[0].message.content:
                    response_content = prompt_response.choices[0].message.content
                    response_content_bits = response_content.split('```')
                    if len(response_content_bits) == 3:
                        csv_string = response_content_bits[1]
                        if csv_string.startswith('csv\n'):
                            csv_string = csv_string.removeprefix('csv\n')
                        try:
                            csv_string_io = StringIO(csv_string)
                            emo_df = pd.read_csv(csv_string_io, sep=",")
                            # print(f'successful emo_df={emo_df}')
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

    # print(f'emo_dict={emo_dict}')
    # print(f'emo_df={emo_df}')
    
    return emo_df, emo_dict


def generate_emotional_sentiment_multi_speaker(text: str) -> tuple[pd.DataFrame, dict]:
    prompt = f"""
    Hi ChatGPT! I'd like your help quantifying the range and intensity of emotions expressed by different characters in a movie scene.

    Before explaining what I mean by "quantifying the range and intensity of emotions," let's examine the movie scene's format. A scene is a series of "lines," with each line spoken by an individual character. Each line begins with the name (in ALL-CAPS) of the character who is speaking it, followed by a colon, then the dialog spoken by the character. Each line is separated from the other lines in the scene by a line break. 
    
    For example, "JOHN: I'm happy to see you." is a line spoken by a character named John, seemingly telling another chracter he is happy to see them. This line might be followed by a line like "JANE: I'm happy to see you too John."

    In many cases, each character in the scene will have more than one line. Although each line is distinct from every other line, all lines spoken by the same character should be thought of as linking back to the same identity, and even if the meaning of the lines are very different, the identity of the speaker is constant - as well as various attributes about the speaker, such as how they feel about another person, what mood they are in, etc.   
    
    Having set that context, I would like your help assessing every character's intensity of emotion across 10 dimensions: (1) Joy, (2) Love, (3) Empathy, (4) Curiosity, (5) Sadness, (6) Anger, (7) Fear, (8) Disgust, (9) Surprise, (10) Confusion. 

    For each of these 10 emotions, and for every character in the scene, please return a decimal value between 0 and 1 indicating the 'score' (or 'weight' / 'intensity') of that emotion for that character. The 'score' should be precise to 2 decimal places.
    
    (Note: there need not be any relationship between individual emotional metric scores. In aggregate, they may all add up to less than 1.0, or their combined total may greatly exceed 1.0 - each emotional metric is independent of the others.)

    In rare cases, more than one person will say the same line at the same time - in those cases, their names might be separated by a plus sign, like "JOHN + JANE: I can't wait!" When this happens, the emotional sentiment of the dialog spoken should be attributed to all characters who have spoken the line together.
     
    Provide the response in CSV format with 4 columns for (1) 'character', (2) 'emotion', (3) 'score', (4) 'explanation' for score. If any column contains quotes or apostrophes, please be sure to escape the text properly. 
    
    Here is the text of the scene:

    {text}
    """

    prompt_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )

    success = False
    if prompt_response.choices:
        if len(prompt_response.choices) > 0:
            if prompt_response.choices[0].message:
                if prompt_response.choices[0].message.content:
                    response_content = prompt_response.choices[0].message.content
                    response_content_bits = response_content.split('```')
                    if len(response_content_bits) == 3:
                        csv_string = response_content_bits[1]
                        if csv_string.startswith('csv\n'):
                            csv_string = csv_string.removeprefix('csv\n')
                        try:
                            csv_string_io = StringIO(csv_string)
                            emo_df = pd.read_csv(csv_string_io, sep=",")
                            # print(f'successful emo_df={emo_df}')
                            success = True
                        except Exception as e:
                            print(f'failure to init StringIO for csv_string={csv_string}')
                            return None, {}
    
    if not success:
        print(f'problem parsing prompt_response={prompt_response}')    
        return None, {}
    
    if emo_df is None or 'emotion' not in emo_df or 'score' not in emo_df:
        print(f'unable to convert `emo_df` to dict after analyzing scene in generate_emotional_sentiment_multi_speaker, original text={text}')
        return None, {}
    
    emo_dict = {}
    slim_emo_df = emo_df[['character', 'emotion', 'score']]
    characters = slim_emo_df['character'].unique()
    if len(characters) == 0:
        print(f'unable to convert `emo_df` to dict after analyzing scene in generate_emotional_sentiment_multi_speaker, `character` column was empty. Original text={text}')
        return emo_df, {}
    for c in characters:
        char_emo_df = slim_emo_df.loc[slim_emo_df['character'] == c]
        emo_dict[c] = dict(zip(char_emo_df.emotion, char_emo_df.score))

    # column name alignment 'character' -> 'speaker'
    emo_df.rename(columns={'character': 'speaker'}, inplace=True)

    # print(f'emo_dict={emo_dict}')
    # print(f'emo_df={emo_df}')
    
    return emo_df, emo_dict


def generate_sentiment(text: str, analyzer: str, multi_speaker: bool = False) -> tuple[pd.DataFrame, dict]:
    if analyzer == 'nltk_pol':
        if multi_speaker:
            print(f'multi_speaker not supported for analyzer={analyzer}')
            return None, {}
        else:
            return generate_polarity_sentiment(text)
    elif analyzer == 'openai_emo':
        if multi_speaker:
            return generate_emotional_sentiment_multi_speaker(text)
        else:
            return generate_emotional_sentiment(text)
    else:
        print(f'Unsupported analyzer={analyzer}')
        return None, {}


# def generate_emotional_sentiments(texts: list) -> tuple[pd.DataFrame, dict]:
#     if len(texts) == 0:
#         return None
    
#     prompt = f"""
#     Hi ChatGPT! I'd like your help quantifying the range and intensity of emotions expressed in snippet of text. When I feed you a text snippet, please assess the degree of emotion in 10 dimensions: (1) Joy, (2) Love, (3) Empathy, (4) Curiosity, (5) Sadness, (6) Anger, (7) Fear, (8) Disgust, (9) Surprise, (10) Confusion. 
    
#     For each of these 10 emotions, please return a decimal value between 0 and 1 indicating the 'score' (or 'weight' / 'intensity') of that emotion. The 'score' should be precise to 2 decimal places.
    
#     Note: there need not be any relationship between the individual emotional metric scores. In aggregate, they may all add up to less than 1.0, or their combined total may greatly exceed 1.0 - each emotional metric is independent of the others. 
    
#     Provide the response in CSV format with 3 columns for (1) 'emotion', (2) 'score', (3) 'explanation' for score. If column (3) contains any quotes or apostrophes, please be sure to escape the text properly. 
#     """
#     message_prompts = [{"role": "user", "content": prompt}]
#     first = True
#     for text in texts:
#         if first:
#             message_prompts.append({"role": "user", "content": f'First text to emotionally anaylze: `{text}`'})
#             first = False
#         else:
#             message_prompts.append({"role": "user", "content": f'Next text to emotionally anaylze: `{text}`'})

#     prompt_response = openai_client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=message_prompts,
#         stream=False,
#     )

#     if prompt_response.choices:
#         if len(prompt_response.choices) > 0:
#             print(f'len(prompt_response.choices)={len(prompt_response.choices)}')
#             print(f'prompt_response.choices={prompt_response.choices}')


#     return None, {}
    
