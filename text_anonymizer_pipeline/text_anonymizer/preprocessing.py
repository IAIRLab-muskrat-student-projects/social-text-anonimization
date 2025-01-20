
import re
from cleantext import clean

# Remove emojis from text
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

# Remove mentions (@username)
def remove_mentions(data):
    return re.sub(r'@\w+', '', data)

# Split hashtags (e.g., #example to " #example")
def split_hashtags(data):
    return re.sub(r'#', ' #', data)

# Remove HTML tags
def remove_html_tags(data):
    return re.sub(r'<.*?>', ' ', data)

# Remove "expand text…" placeholder in social media
def remove_expand(data):
    return re.sub(r'expand text…', '', data)

# Comprehensive cleaning function
def apply_clean(doc, vk_preprocess=False):
    doc = remove_html_tags(doc)
    doc = clean(doc, fix_unicode=True, lower=True, no_line_breaks=True,
                no_urls=True, no_emails=True, no_phone_numbers=True, 
                no_currency_symbols=True, lang="en")
    doc = remove_emojis(doc)
    if vk_preprocess:
        doc = remove_mentions(doc)
        doc = split_hashtags(doc)
        doc = remove_expand(doc)
    return doc
        