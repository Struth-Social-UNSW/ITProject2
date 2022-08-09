import spacy

nlp = spacy.load('en_core_web_sm')

sample_text = "testing the tokenizer"

tokens_found = []
for tokens in sample_text:
    tokens_found.append(items.text)

print(tokens_found)