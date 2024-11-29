import re

import nltk
from cleantext import clean

nltk.download('stopwords')
nltk.download('wordnet')

eng_stopwords = stopwords.words('english')

CORES = 10

SEED = 42
TRAIN_DOC_COUNT = 10000
TEST_DOC_COUNT = 1000
AUTHOR_COUNT = 100

# убираю различные эмодзи
def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

# убираю упоминания в инсте
def remove_mentions(data):
    return re.sub(r'(@[\w_]+)|(\[.*\](,? )?)', '', data)

# хэштеги часто пишут подряд без пробелов
def split_hashtags(data):
    return re.sub(r'#', ' #', data)

# просто убираю лишние символы ()_
def remove_extra_symbols(data):
    return re.sub(r'[\(\)_]+', '', data)

# в вк есть html теги, их тоже надо убрать
def remove_html_tags(data):
    return re.sub(r'<.*?>', ' ', data)

# в текстах из вк часто присуствует expand text, скорее всего, 
# просто не удалось спарсить остальной текст
def remove_expand(data):
    return re.sub(r'expand text…', '', data)

# здесь все функции для чистки данных собраны в одном месте, 
# чтобы можно было применять к тексту, подходит для метода pd.DataFrame.apply
def apply_clean(doc, vk_preprocess=False):
    doc = remove_html_tags(doc)
    # clean из библиотеки cleantext, тоже сильно помогает
    doc = clean(doc,
                fix_unicode=True,               # fix various unicode errors
                to_ascii=False,                  # transliterate to closest ASCII representation
                lower=True,                     # lowercase text
                no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
                no_urls=True,                  # replace all URLs with a special token
                no_emails=True,                # replace all email addresses with a special token
                no_phone_numbers=True,         # replace all phone numbers with a special token
                no_numbers=False,               # replace all numbers with a special token
                no_digits=False,                # replace all digits with a special token
                no_currency_symbols=True,      # replace all currency symbols with a special token
                no_punct=False,                 # remove punctuations
                replace_with_punct="",          # instead of removing punctuations you may replace them
                replace_with_url="",
                replace_with_email="",
                replace_with_phone_number="",
                replace_with_currency_symbol="",
                lang="en"                       # set to 'de' for German special handling
               )
    doc = remove_emojis(doc)
    if vk_preprocess:
        doc = remove_mentions(doc)
        doc = split_hashtags(doc)
        doc = remove_extra_symbols(doc)
        doc = remove_expand(doc)
    
    return doc

# разделение текста и хэштегов друг от друга
def get_text_and_hashtags(row, text_col='text'):
    text = row[text_col]
    if len(text) == 0:
        return ''
    return re.sub('#\w+', '', text).strip(), ' '.join(re.findall('#\w+', text))