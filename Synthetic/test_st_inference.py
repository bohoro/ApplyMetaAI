from sentence_transformers import SentenceTransformer, util

model_sepc = "all-distilroberta-v1"
model = SentenceTransformer(model_sepc)


def compare_sentences(sent1, sent2):
    """takes 2 sentences, returns the Cosine Similarity"""
    # Sentences are encoded by calling model.encode()
    emb1 = model.encode(sent1)
    emb2 = model.encode(sent2)
    return util.cos_sim(emb1, emb2)


def main():
    cos_sim = compare_sentences(
        "This is a red cat with a hat.", "Have you seen my red cat?"
    )
    print(f"Cosine-Similarity: {cos_sim}")


if __name__ == "__main__":
    main()
