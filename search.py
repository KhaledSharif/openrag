import ollama, sys, chromadb
from ollama import Options
from utilities import getconfig
import dspy
import re


# ref: https://github.com/stanfordnlp/dspy/blob/main/examples/qa/hotpot/hotpotqa_with_MIPRO.ipynb
class ReturnRankedDocuments(dspy.Signature):
    """Given a question we are trying to answer and a list of passages,
    return a comma separated list of the indices associated with each passage.
    These indices should be ordered by helpfulness in answering the question,
    with most helpful passage index first, and the least helpful passage index last."""

    question = dspy.InputField(
        desc="The question we are trying to answer using passages in the context"
    )
    context = dspy.InputField(desc="List of potentially related passages")
    ranking = dspy.OutputField(
        desc="""
        A comma separated list of numbers corresponding to passage indices ordered 
        by helpfulness in answering the question, with most helpful passage index first, 
        and the least helpful passage index last
        """
    )


generate_ranking = dspy.ChainOfThoughtWithHint(ReturnRankedDocuments)


lm_provider = dspy.OllamaLocal(
    model=getconfig()["mainmodel"],
    max_tokens=int(getconfig()["npredict"]),
    num_ctx=int(getconfig()["ncontext"]),
)

dspy.settings.configure(lm=lm_provider)

ollama_options = Options(
    num_predict=int(getconfig()["npredict"]),
    seed=42,
    temperature=0.0,
    num_ctx=int(getconfig()["ncontext"]),
)


embedmodel = getconfig()["embedmodel"]
mainmodel = getconfig()["mainmodel"]

chroma = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma.get_or_create_collection(getconfig()["name"])

query = " ".join(sys.argv[1:])
queryembed = ollama.embeddings(model=embedmodel, prompt=query)["embedding"]

n_docs = int(getconfig()["ndocs"])

assert n_docs >= 1

docs = collection.query(query_embeddings=[queryembed], n_results=n_docs)["documents"][0]
docs_joined = "\n\n\n".join(
    [f"<passage_{i}>{d}</passage_{i}>" for i, d in enumerate(docs)]
)

print(docs_joined)

n_hops = int(getconfig()["nhops"])

for hop in range(n_hops):
    print(f"HOP {hop+1} of {n_hops}")

    # Get the most important indices, ranked
    indices_string = generate_ranking(
        question=query,
        context=docs_joined,
        hint=f"""
        Return a comma seperated list of indices from 0 to {len(docs)-1} corresponding to 
        passage indices ordered by helpfulness in answering the question, 
        with most helpful passage index first, and the least helpful passage index last
        """,
    ).ranking

    # indices = [int(num) for num in re.findall(r"\d+", indices_string)]

    print(indices_string)

    exit()


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
