from underthesea import word_tokenize

def segment_text(text):
    """
    Segment Vietnamese text using underthesea.
    """
    try:
        return word_tokenize(text, format='text')
    except Exception:
        return ""
