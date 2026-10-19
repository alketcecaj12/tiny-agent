"""Chapter 4 - Long-term memory with embeddings and RAG.

Needs an embedding model, e.g. `ollama pull embeddinggemma`.
"""

from tinyagent import config
from tinyagent.agent import TinyAgent
from tinyagent.llm import LLM, EmbeddingModel
from tinyagent.memory import RAGMemory

DOCUMENTS = [
    "Sarah works as a marine biologist studying coral reefs.",
    "Sarah lives in Lisbon, Portugal.",
    "Sarah's favorite hobby is rock climbing.",
    "Sarah favorite animals are flamingos.",
    "Sarah speaks fluent Spanish and Portuguese.",
    "Ilse is a software engineer at a renewable energy startup.",
    "Ilse lives in Amsterdam, the Netherlands.",
    "Ilse plays the cello in a local string quartet.",
    "Ilse's favorite author is Brandon Sanderson.",
    "Ilse's favorite animals are dolphins.",
]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5))


def main():
    embedding_model = EmbeddingModel()
    print(f"Embedding size: {len(embedding_model.embed('Dolphins are amazing!'))}\n")

    # Semantic similarity
    a = embedding_model.embed("I love flamingos.")
    b = embedding_model.embed("Dolphins use echolocation.")
    c = embedding_model.embed("Flamingos are pink birds.")
    print(f"'I love flamingos.' vs 'Dolphins use echolocation.': {cosine(a, b):.3f}")
    print(f"'I love flamingos.' vs 'Flamingos are pink birds.':  {cosine(a, c):.3f}\n")

    # RAG memory
    llm = LLM(model=config.model("prompt"))
    agent = TinyAgent(llm=llm, memory=RAGMemory(embedding_model, DOCUMENTS))
    print(agent.run("What is Sarah's favorite animal?"), "\n")

    print("== Memory contents")
    for message in agent.memory.get_messages():
        print(f"{message['role']}: {message['content']}\n")


if __name__ == "__main__":
    main()
