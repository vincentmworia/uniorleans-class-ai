# ===========================================
# Word2Vec – Skip-Gram on Toy "Royal" Corpus
# With Extensions A–D
# ===========================================

import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# FLAGS: set to False if you don't want to run a given extension
# -------------------------------------------------------------------
RUN_BASE        = True
RUN_EXTENSION_A = True
RUN_EXTENSION_B = True
RUN_EXTENSION_C = True
RUN_EXTENSION_D = True

# -------------------------------------------------------------------
# 1) BASE CORPUS
# -------------------------------------------------------------------
base_corpus = [
    "the king and the queen live in the castle",
    "the man and the woman walk in the village",
    "the prince is the son of the king",
    "the princess is the daughter of the queen",
    "the woman loves the child",
    "the man works in the castle",
    "the queen loves the prince",
    "the king rules the kingdom",
    "the princess loves the child",
    "the prince and the princess walk together"
]

# Helper function to train and inspect a model ----------------------
def train_and_inspect(corpus, vector_size=10, window=2, sg=1, label=""):
    print("\n" + "=" * 60)
    print(f"Training model: {label}")
    print("=" * 60)

    sentences = [s.split() for s in corpus]

    model = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=1,
        sg=sg,
        epochs=100
    )

    # Similar words
    for w in ["king", "queen", "man", "woman"]:
        print(f"\nMost similar to '{w}':")
        print(model.wv.most_similar(w))

    # Analogy
    result = model.wv.most_similar(
        positive=["king", "woman"],
        negative=["man"],
        topn=1
    )
    print("\nAnalogy: king - man + woman ≈", result[0][0])

    # One example embedding
    print("\nEmbedding for 'king':")
    print(model.wv["king"])

    return model


# -------------------------------------------------------------------
# BASE MODEL (Skip-Gram, window=2)
# -------------------------------------------------------------------
if RUN_BASE:
    model = train_and_inspect(
        base_corpus,
        vector_size=10,
        window=2,
        sg=1,
        label="BASE: Skip-Gram, window=2, original corpus"
    )

    # TSNE for base model
    words = ["king", "queen", "man", "woman",
             "prince", "princess", "child", "castle", "village"]
    vectors = np.array([model.wv[w] for w in words])

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    X_2d = tsne.fit_transform(vectors)

    plt.figure(figsize=(6, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1])
    for i, w in enumerate(words):
        plt.text(X_2d[i, 0] + 0.02, X_2d[i, 1] + 0.02, w, fontsize=12)
    plt.title("TSNE – Base Model (Skip-Gram, window=2)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# EXTENSION A: Add new sentences to the corpus
# -------------------------------------------------------------------
extended_corpus = base_corpus[:]  # start from base

if RUN_EXTENSION_A:
    extra_sentences = [
        "the king leads the army",
        "the woman teaches the child at school",
        "the man and the king ride horses"
    ]
    extended_corpus.extend(extra_sentences)

    model = train_and_inspect(
        extended_corpus,
        vector_size=10,
        window=2,
        sg=1,
        label="EXT A: Skip-Gram, window=2, extended corpus"
    )


# -------------------------------------------------------------------
# EXTENSION B: Increase context window to 4
# -------------------------------------------------------------------
if RUN_EXTENSION_B:
    model = train_and_inspect(
        extended_corpus,
        vector_size=10,
        window=4,
        sg=1,
        label="EXT B: Skip-Gram, window=4, extended corpus"
    )


# -------------------------------------------------------------------
# EXTENSION C: Train CBOW model (sg = 0)
# -------------------------------------------------------------------
if RUN_EXTENSION_C:
    model = train_and_inspect(
        extended_corpus,
        vector_size=10,
        window=4,
        sg=0,
        label="EXT C: CBOW, window=4, extended corpus"
    )


# -------------------------------------------------------------------
# EXTENSION D: Add 3 new words to TSNE visualization
# (uses the LAST trained model, usually CBOW from Extension C)
# -------------------------------------------------------------------
if RUN_EXTENSION_D:
    # words we know are in the extended corpus
    tsne_words = [
        "king", "queen", "man", "woman",
        "prince", "princess", "child", "castle", "village",
        "army", "school", "horses"
    ]

    # make sure all words exist in vocab
    tsne_words = [w for w in tsne_words if w in model.wv.key_to_index]

    vectors = np.array([model.wv[w] for w in tsne_words])

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    X_2d = tsne.fit_transform(vectors)

    plt.figure(figsize=(6, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1])
    for i, w in enumerate(tsne_words):
        plt.text(X_2d[i, 0] + 0.02, X_2d[i, 1] + 0.02, w, fontsize=11)
    plt.title("TSNE – Extension D (extra words added)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.show()
