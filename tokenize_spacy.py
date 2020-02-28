import spacy

def tokenize_de(text):
    """ Tokenize the German text into a list of tokens """
    spacy_de=spacy.load('de')
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    spacy_en=spacy.load('en')
    return [tok.text for tok in spacy_en.tokenizer(text)]
