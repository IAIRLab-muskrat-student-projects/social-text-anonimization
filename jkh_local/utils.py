import re
import requests
from typing import Union

from cleantext import clean


def safe_str(s: Union[str, None, float]):
    return s if type(s) == str else ''


re_street_kinds = '(((((пр)|(ул)|(пл)|(пер)|(\d\-[йя])|(наб(\.р(еки)?)?)|(ш))(([\.\s]\s*)|(,)))|(((улиц[аы])|(линия)|(дорог[аи])|(проспекта?)|(шоссе)|(переулока?)|(площадь)|(набережная)|(бульвар))\s*)))'
re_street_name = '[\d\w\s-]+'

group_street = 'street'
re_street = f'(?P<{group_street}>(({re_street_kinds}{re_street_name})|({re_street_name}{re_street_kinds}))([\d\w\s\.-])*)'

def find_street(address: str):
    m = re.search(re_street, safe_str(address))
    if m is not None:
        m = m.group(group_street)
    return m


re_house_number = '((\d)+(\\\d+)?(\w)*)'
re_house_type = '((дом[\s\-]+)|(д[\s\.\-]+))'
re_litera = '(((литера\s)|(лит\.)|(л\.))\s*\w)'
re_corpus = '(((корпус\s)|(корп\.)|(к\.))\s*\d+)'
group_house = 'house'
re_house = f'(?P<{group_house}>({re_house_type}{re_house_number}))'
re_split = '(\s*,\s*)?'
group_house_full = 'house_full'
re_house_full = f'(?P<{group_house_full}>({re_house}{re_split}((({re_litera}{re_split})?({re_corpus})?)|(({re_corpus}{re_split})?({re_litera})?))))'


group_street = 'street'
re_street = f'(?P<{group_street}>(({re_street_kinds}{re_street_name})|({re_street_name}{re_street_kinds}))([\d\w\s\.-])*)'


def find_house(address: str):
    m = re.search(re_house_full, safe_str(address))
    if m is not None:
        m = m.group(group_house_full)
    return m


def get_url_content(url: str):
    res = requests.get(url)
    return res.json()


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

def remove_mentions(data):
    return re.sub(r'@[\w_]+', '', data)

def split_hashtags(data):
    return re.sub(r'#', ' #', data)

def remove_extra_symbols(data):
    return re.sub(r'[\(\)_"\'«»]+', '', data)

def remove_html_tags(data):
    return re.sub(r'<.*?>', ' ', data)

def remove_expand(data):
    return re.sub(r'expand text…', '', data)

def apply_clean(doc):
    doc = remove_html_tags(doc)
    doc = clean(doc,
                fix_unicode=True,               # fix various unicode errors
                to_ascii=False,                  # transliterate to closest ASCII representation
                lower=False,                     # lowercase text
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
                replace_with_currency_symbol="",
                lang="en"                       # set to 'de' for German special handling
               )
    doc = remove_emojis(doc)
    doc = remove_mentions(doc)
    doc = split_hashtags(doc)
    doc = remove_extra_symbols(doc)
    doc = remove_expand(doc)
    
    return doc