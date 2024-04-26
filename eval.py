import ollama, chromadb
from ollama import Options
from utilities import getconfig


ollama_options = Options(num_predict=1_000, seed=42, temperature=0.0, num_ctx=8_000)

embedmodel = getconfig()["embedmodel"]

with open(getconfig()["questions"]) as f:
    q_lines = f.readlines()


embedmodel = getconfig()["embedmodel"]

evalmodels = getconfig()["evalmodels"].split("|")

chroma = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma.get_or_create_collection(getconfig()["name"])

responses = {}


for model in evalmodels:
    print("=" * 40)
    print(f"RESPONSE FROM {model.upper()}")
    print("=" * 40)

    for question in q_lines:

        if not (question in responses):
            responses[question] = {}

        responses[question][model] = ""

        queryembed = ollama.embeddings(model=embedmodel, prompt=question)["embedding"]

        n_docs = int(getconfig()["ndocs"])

        assert n_docs >= 1

        docs = collection.query(query_embeddings=[queryembed], n_results=n_docs)[
            "documents"
        ][0]

        docs = "\n\n".join(docs)

        modelquery = f"""
        {docs}
        \n\n\n 
        Provide an answer to the following question using the previous text as a resource. 
        \n\n\n 
        Question: 
        "{question}" 
        \n\n\n 
        Answer:
        """

        print("\n", "-" * 40, "\n")
        print("Q:", question, "\n")

        stream = ollama.generate(
            model=model, prompt=modelquery, stream=True, options=ollama_options
        )

        for chunk in stream:
            if chunk["response"]:

                responses[question][model] += chunk["response"]

                print(chunk["response"], end="", flush=True)

        print("")
        print("-" * 40)

    print("=" * 40)
