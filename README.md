# ZZSN 24L - chatbot project

## Wstęp
Poniższa instrukcja opisuje:
1. Przygotowanie środowiska i uruchamianie aplikacji
3. Funkcje i przeznaczenie poszczególnych plików

## Przygotowanie środowiska i uruchamianie aplikacji
1. Utwórz środowisko wirtualne (conda, venv, etc.), np.: `python -m venv zzsn_chatbot`
2. Aktywuj środowisko wirtualne (w zależności od systemu operacyjnego):
    - Windows: `.\aerial_images\Scripts\activate`
    - Linux: `source aerial_images/bin/activate`
3. Pobierz i zainstaluj bibliotekę `torch` zgodnie z instrukcją na stronie [pytorch.org](https://pytorch.org/get-started/locally/)
4. Zainstaluj pozostałe wymagane biblioteki: `pip install -r requirements.txt`
5. W celu uruchomienia aplikacji wykonaj:
    * `python src/chatbots/main_rag.py` - uruchomienie chatbota z RAG
    * `python src/chatbots/main_no_rag.py` - uruchomienie chatbota bez RAG
6. Interfejs powinien być widoczny na `[127.0.0.](http://127.0.0.1:7860)

## Kod i funkcje poszczególnych plików
Ogólnie repozytorium jest zorganizowane w taki sposób, że katalog `src` zawiera kod źródłowy aplikacji wymienionych w raporcie, katalog `notebooks` zawiera notebooki z eksperymentami, które demonstrują wypróbowane podejścia (nie wszystkie działające), a katalog `appendix` zawiera dodatkowe pliki, wymienione w raporcie. Opis poszczególnych sub-katalogów:

* `appendix` - pliki wymienione w raporcie, ale nie będące częścią kodu aplikacji
* `notebooks` - notebooki z eksperymentami
    * `llama_cpp` - próby związane z integracją LLama Cpp, szczególnie modeli kwantyzowanych, w formacie GGUF
    * `misc` - pozostałe próby, demonstracje modeli 'zagnieżdżających' (*embeddings models*)
* `src` - pliki będące częścią kodu aplikacji
    * `scripts` - skrypty użyte do niektórych testów, opisanych w raporcie
    * `chatbots` - działające chatboty, z RAG lub bez