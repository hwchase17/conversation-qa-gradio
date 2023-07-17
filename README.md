# LangChain <> Gradio Custom QA Over Docs Bot

This repo shows how to create a Chatbot over your documents using LangChain and Gradio.
Importantly, this has an emphasis on using some of the lower level components of LangChain rather than a predefined chain.

This also uses:
- Cohere for embeddings
- ChromaDB for a vectorstore
- OpenAI for a text generation model

## Setup

To setup, please install requirements with `pip install -r requirements.txt`

Then, set various environment variables:

```shell
export OPENAI_API_KEY=...
export COHERE_API_KEY=
```

## Ingest

First, we need to ingest data.
For this example, we will work with a state of the union address (`state_of_the_union.txt`).
You can modify the code in `ingest.py` to ingest anything you want.
To ingest, run `python ingest.py`

## Chat

Now we can chat with this data! In order to do that, run `python app.py`.
This will spin up a Gradio application that you can chat with in the frontend.
For details on how to customize the chatbot, see the code in `app.py`