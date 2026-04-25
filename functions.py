from transformers import pipeline

# f do analizy uczuc
def analyze_sentiment(text):
    classifier = pipeline("sentiment-analysis")
    return classifier(text)

# f do tłumaczenia na niemiecki
def translate_text(text):
    translator = pipeline(
        "translation_en_to_de",
        model="Helsinki-NLP/opus-mt-en-de"
    )
    result = translator(text)
    return result[0]["translation_text"]
