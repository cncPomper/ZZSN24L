# ZZSN 24L - chatbot project

## TODO
- [x] wstępna działająca wersja chatbota
- [ ] dodanie możliwości wgrania pliku przez użytkownika (np. dataset .csv) i przetworzenia go przez chatbota, a następnie rozmowy na temat
- [ ] wybór / testy LLM
    - [ ] llama2
    - [ ] mistral
    - [ ] któryś z polskich
- [ ] wersja, która mówi z sensem
- [ ] dodać element *RAG-like*, żeby czathob rozmawiał w odniesieniu do danych
- [ ] dokumentacja projektu

## Setup
* `pip install -r requirements.txt`
* run bunch of chatbots or 'chatbots' from gradio docs: 
    1. `python src/{bot_name}.py`
    2. open browser and go to `http://localhost:7860/`
* available options are in [src](./src/) folder:
    * `random_responder.py`
    * `draft_responder.py` (this one doesn't work yet, it's a draft for now)
    * `streaming_responder.py`
    * `simple_custom_responder.py`
    * `actual_llm_responder.py` (this use uses models from hf, so getting them may take some time, depending on your internet connection)
    * `multimodal_but_stupid_responder.py`

## Development
* na podstawie tutoriala z yt
    - GradioModel - data model, danych, które będą przesyłane do chatbota
    - CustomComponents - pisanie własnych komponentów

## Dane
Datasety są opisane w [data](./data/).

## Wyniki
* halucynacje mocno so far 
![part-0](./assets/results-0.png)
![part-1](./assets/results-1.png)

## Źródła
* [gradio docs general](https://www.gradio.app/docs/gradio/chatbot)
* [gradio docs fast](https://www.gradio.app/guides/creating-a-chatbot-fast)
* [gradio docs customization](https://www.gradio.app/guides/creating-a-custom-chatbot-with-blocks)
* [gradio docs tutorial multimodal](https://www.gradio.app/guides/multimodal-chatbot-part1)
* [medium](https://medium.com/@anu.surabhi1980/building-a-simple-chatbot-with-transformers-and-gradio-c7913c21217f)
* [yt tutorial on multimodal chatbot](https://www.youtube.com/watch?v=IVJkOHTBPn0&ab_channel=HuggingFace)
* [llm file input some blogpost](https://shelf.io/blog/understanding-the-influence-of-llm-inputs-on-outputs/)
* [llm file input medium langchain](https://medium.com/@hamzafergougui/speak-to-your-data-using-langchain-and-llms-78afb42d4c36)
