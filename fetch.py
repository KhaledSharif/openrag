import chromadb, time
import ollama
from utilities import get_filename, readtext, getconfig
import re
import dspy
import html2text
from textwrap import wrap
from tqdm import tqdm


class Fetcher:
    def __init__(self) -> None:
        self.text_maker = html2text.HTML2Text()
        self.text_maker.ignore_links = True
        self.text_maker.ignore_images = True
        self.text_maker.ignore_emphasis = True
        self.text_maker.ignore_mailto_links = True
        self.text_maker.ignore_tables = False
        self.text_maker.bypass_tables = False
        self.text_maker.body_width = 150

        self.chroma = chromadb.HttpClient(host="localhost", port=8000)


def clean_string(input_string):
    # Remove non-ASCII characters
    ascii_string = re.sub(r"[^\x00-\x7F]+", "", input_string)

    # Remove extra whitespace
    clean_string = re.sub(r"\n{2,}", "\n", ascii_string)

    return clean_string


collection_name = getconfig()["name"]


class ExtractText(dspy.Signature):
    """Extract the useful text from the information provided in the document"""

    document = dspy.InputField(
        desc="""
        provided document which contains information to use as a resource
        when extracting the useful text
        """
    )
    text = dspy.OutputField(
        desc="""
        useful text extracted from the document containing the main information
        """
    )


lm_provider = dspy.OllamaLocal(
    model=getconfig()["chunkmodel"],
    max_tokens=int(getconfig()["npredict"]),
    num_ctx=int(getconfig()["ncontext"]),
)

dspy.settings.configure(lm=lm_provider)

gen_chunks = dspy.ChainOfThoughtWithHint(ExtractText)


def generate_chunks(chunk):
    c = gen_chunks(
        document=chunk,
        hint="""
        extract useful text from the document
        """,
    )
    return c.text


if __name__ == "__main__":
    fetcher = Fetcher()

    if any(
        collection.name == collection_name
        for collection in fetcher.chroma.list_collections()
    ):
        print("deleting collection")
        fetcher.chroma.delete_collection(collection_name)

    collection = fetcher.chroma.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    embedmodel = getconfig()["embedmodel"]

    starttime = time.time()

    with open(getconfig()["sources"]) as f:
        lines = f.readlines()

    for line_index, filename in enumerate(lines):
        print("\n", "=" * 40, "\n")
        print(filename, "\n")

        chunks = fetcher.text_maker.handle(readtext(filename).prettify())

        step_size = int(getconfig()["chunksize"])
        step_overlap = step_size // 10
        for chunk_index in tqdm(range(0, len(chunks), step_size - step_overlap)):
            chunk_end = min(len(chunks), chunk_index + step_size)

            chunk = chunks[chunk_index:chunk_end]
            chunk = clean_string(chunk.strip()).strip()

            if len(chunk) < 1_000:
                print("chunk too small, skipping")
                continue

            chunk_processed = generate_chunks(chunk)

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
