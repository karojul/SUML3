import streamlit as st
import pandas as pd
import torch
from transformers import pipeline
from functions import analyze_sentiment, translate_text

import time
# import matplotlib as plt
import os

# zaczynamy od zaimportowania bibliotek

# st.info('Gratulacje! Z powodzeniem uruchomiłeś aplikację')
# .streamlit jest wykorzystywany do tworzenia aplikacji
# z tego powodu dobrą praktyką jest informowanie użytkownika o postępie, błędach, etc.

# Inne przykłady do wypróbowania:
# st.balloons() # animowane balony ;)
# st.error('Błąd!') # wyświetla informację o błędzie
# st.warning('Ostrzeżenie, działa, ale chyba tak sobie...')
# st.info('Informacja...')
# st.success('Udało się!')

# st.spinner()
# with st.spinner(text='Pracuję...'):
    # time.sleep(2)
    # st.success('Done')
# możemy dzięki temu "ukryć" późniejsze ładowanie aplikacji
st.set_page_config(
    page_icon="🗣️",
    layout="centered"
)

st.title('Analiza i tłumaczenie tekstów')

st.header('Instrukcja')
st.subheader('Jak działa tłumacz')
st.text('Skorzystaj z naszego narzędzia do analizy emocji w tekście lub przetłumaczenie zdań z angielskiego na niemiecki! ')
# st.write('Streamlit jest biblioteką pozwalającą na uruchomienie modeli uczenia maszynowego.')

# df = pd.read_csv("../../Downloads/DSP_4.csv", sep =';')
# st.dataframe(df)
# musimy tylko pamiętać o właściwym określeniu separatora (w tym wypadku to średnik)
# masz problem z otworzeniem pliku? sprawdź w jakim katalogu pracujesz i dodaj tam plik (albo co bardziej korzystne - zmień katalog pracy)
# os.getcwd() # pokaż bieżący katalog
# os.chdir("") # zmiana katalogu

st.header('Przetwarzanie języka naturalnego')

def load_sentiment_model():
    return pipeline("sentiment-analysis")

def load_translation_model():
    return pipeline(
        "text2text-generation",
        model="Helsinki-NLP/opus-mt-en-de",
    )

option = st.selectbox(
    "Opcje",
    [
        "Wydźwięk emocjonalny tekstu (eng)",
        "Tłumaczenie angielskiego na niemiecki",
    ],
)

if option == "Wydźwięk emocjonalny tekstu (eng)":
    text = st.text_area(label="Wpisz tekst")

    if text:
        with st.spinner("Analizuję tekst..."):
            answer = analyze_sentiment(text)

        st.success("Analiza zakończona!")
        st.write(answer)

elif option == "Tłumaczenie angielskiego na niemiecki":
    text = st.text_area(label="Wpisz tekst po angielsku")
    st.image("EnDe.png")

    if text:
        with st.spinner("Tłumaczę tekst..."):
            result = translate_text(text)

        st.balloons()
        st.success("Tłumaczenie gotowe!")
        st.info(result)


st.subheader('Zadanie do wykonania')
st.write('Wykorzystaj Huggin Face do stworzenia swojej własnej aplikacji tłumaczącej tekst z języka angielskiego na język niemiecki. '
         'Zmodyfikuj powyższy kod dodając do niego kolejną opcję, tj. tłumaczenie tekstu. Informacje potrzebne do zmodyfikowania kodu '
         'znajdziesz na stronie Huggin Face - https://huggingface.co/docs/transformers/index')
# st.write('🐞 Dodaj właściwy tytuł do swojej aplikacji, może jakieś grafiki?')
# st.write('🐞 Dodaj krótką instrukcję i napisz do czego służy aplikacja')
# st.write('🐞 Wpłyń na user experience, dodaj informacje o ładowaniu, sukcesie, błędzie, itd.')
# st.write('🐞 Na końcu umieść swój numer indeksu')
# st.write('🐞 Stwórz nowe repozytorium na GitHub, dodaj do niego swoją aplikację, plik z wymaganiami (requirements.txt)')
# st.write('🐞 Udostępnij stworzoną przez siebie aplikację (https://share.streamlit.io) a link prześlij do prowadzącego')

st.divider()
st.caption("Numer indeksu: s26414")
