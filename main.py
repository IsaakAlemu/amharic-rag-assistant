from sentence_transformers import SentenceTransformer
from groq import Groq

from config import get_settings
from src.logging_config import setup_logging
from src.pipeline import answer_question, load_vector_collection


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    client = Groq(api_key=settings.groq_api_key)
    embed_model = SentenceTransformer(settings.embed_model)
    collection = load_vector_collection(settings, embed_model)

    query = input("Ask a question: ")

    result = answer_question(
        query,
        client=client,
        embed_model=embed_model,
        collection=collection,
        settings=settings,
    )

    if result.error:
        print(f"\nError: {result.error}")
        return

    print(f"\nRetrieval time: {result.timings_ms.get('retrieve', 0) / 1000:.2f}s")
    print(f"Generation time: {result.timings_ms.get('generate', 0) / 1000:.2f}s")
    print(f"Total: {result.timings_ms.get('total', 0) / 1000:.2f}s")

    print("\nAnswer:")
    print(result.answer)

    if result.sources:
        print("\nSources:")
        for source in result.sources:
            print(
                f"  [{source['rank']}] Document {source['document_id']} "
                f"(distance={source['distance']:.4f})"
            )


if __name__ == "__main__":
    main()
