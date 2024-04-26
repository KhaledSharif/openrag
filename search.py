import ollama, sys, chromadb
from ollama import Options
from utilities import getconfig

ollama_options = Options(num_predict=1_000, seed=42, temperature=0.0, num_ctx=8_000)

embedmodel = getconfig()["embedmodel"]
mainmodel = getconfig()["mainmodel"]
chroma = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma.get_or_create_collection(getconfig()["name"])

query = " ".join(sys.argv[1:])
queryembed = ollama.embeddings(model=embedmodel, prompt=query)["embedding"]

n_docs = int(getconfig()["ndocs"])

assert n_docs >= 1

docs = collection.query(query_embeddings=[queryembed], n_results=n_docs)["documents"][0]

docs = "\n\n".join(docs)

modelquery = f"""
{docs}
\n\n\n 
Provide an answer to the following question using the previous text as a resource. 
\n\n\n 
Question: 
"{query}" 
\n\n\n 
Answer:
"""

print("=" * 40)
print("PROMPT")
print("=" * 40)
print(modelquery)
print("=" * 40)

print("\n")

stream = ollama.generate(
    model=mainmodel, prompt=modelquery, stream=True, options=ollama_options
)

print("=" * 40)
print("RESPONSE")
print("=" * 40)
for chunk in stream:
    if chunk["response"]:
        print(chunk["response"], end="", flush=True)
print("")
print("=" * 40)
