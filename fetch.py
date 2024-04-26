import chromadb, time
import ollama
from utilities import get_filename, readtext, getconfig
import re
import dspy
import os
import glob
import html2text
from textwrap import wrap
from tqdm import tqdm


text_maker = html2text.HTML2Text()
text_maker.ignore_links = True
text_maker.ignore_images = True
text_maker.ignore_emphasis = True
text_maker.ignore_mailto_links = True
text_maker.ignore_tables = False
text_maker.bypass_tables = False
text_maker.body_width = 150


class ExtractText(dspy.Signature):
    """Extract the useful text from the information provided in the document"""

    document = dspy.InputField(
        desc="provided document which contains information to use as a resource when extracting the useful text"
    )
    text = dspy.OutputField(
        desc="useful text extracted from the document containing the main information"
    )

lm_provider = dspy.OllamaLocal(
    model=getconfig()["chunkmodel"], max_tokens=1_000, num_ctx=4_000
)

dspy.settings.configure(lm=lm_provider)

gen_chunks = dspy.ChainOfThoughtWithHint(ExtractText)


def clean_string(input_string):
    # Remove non-ASCII characters
    ascii_string = re.sub(r"[^\x00-\x7F]+", "", input_string)

    # Remove extra whitespace
    clean_string = re.sub(r"\n{2,}", "\n", ascii_string)

    return clean_string


collection_name = getconfig()["name"]

chroma = chromadb.HttpClient(host="localhost", port=8000)

if any(collection.name == collection_name for collection in chroma.list_collections()):
    print("deleting collection")
    chroma.delete_collection(collection_name)
    
collection = chroma.get_or_create_collection(
    name=collection_name, metadata={"hnsw:space": "cosine"}
)

embedmodel = getconfig()["embedmodel"]
starttime = time.time()
with open(getconfig()["sources"]) as f:
    lines = f.readlines()


for line_index, filename in enumerate(lines):
    print("\n", "=" * 40, "\n")
    print(filename, "\n")
    
    content_path = get_filename(filename)
    
    if os.path.exists(content_path) and os.path.isfile(content_path):
        print("skipping!")
        print("\n", "=" * 40, "\n")
        continue

    soup = readtext(filename).prettify()

    chunks = text_maker.handle(soup)

    step_size = 30_000
    step_overlap = step_size // 10
    for chunk_index in tqdm(range(0, len(chunks), step_size - step_overlap)):
        chunk_end = min(len(chunks), chunk_index + step_size)

        chunk = chunks[chunk_index:chunk_end]
        chunk = clean_string(chunk.strip()).strip()

        if len(chunk) < 1_000:
            continue

        try:
            chunk_processed = gen_chunks(document=chunk, hint="extract useful text from the document").text
        except TypeError as e:
            print("warning type error during dspy extraction")
            continue
        
        chunk_processed = "\n".join(wrap(chunk_processed, width=150))

        with open(
            f"sentences/line_{line_index}_step_{step_size}_chunk_{chunk_index}.txt",
            "w",
        ) as senf:
            senf.write("=" * 40 + "\n")
            senf.write("INPUT" + "\n")
            senf.write("=" * 40 + "\n")
            senf.write(chunk)

            senf.write("\n\n")

            senf.write("=" * 40 + "\n")
            senf.write("OUTPUT" + "\n")
            senf.write("=" * 40 + "\n")
            senf.write(chunk_processed)
            
            senf.write("\n\n")
            

        embed = ollama.embeddings(model=embedmodel, prompt=chunk_processed)["embedding"]
        collection.add(
            [filename + f"l{line_index}_s{step_size}_c{chunk_index}"],
            [embed],
            documents=[chunk_processed],
            metadatas={"source": filename},
        )

print("--- %s seconds ---" % (time.time() - starttime))
