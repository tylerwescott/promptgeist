import argparse

import numpy as np
import torch
from diffusers import StableDiffusionPipeline

import spacy
from tqdm import tqdm

import nltk
from nltk.corpus import words

# Ensure the NLTK English word list is available
try:
    nltk.data.find("corpora/words")
except LookupError:
    nltk.download("words")


POS_TAGS = {"NOUN", "PROPN", "ADJ"}
MIN_TOKEN_LENGTH = 3


def load_pipelines(model_id: str):
    # Load the Stable Diffusion pipeline and get available text encoders
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    encoders = []

    if hasattr(pipe, "text_encoder") and hasattr(pipe, "tokenizer"):
        encoders.append(("text_encoder", pipe.tokenizer, pipe.text_encoder))

    if hasattr(pipe, "text_encoder_2") and hasattr(pipe, "tokenizer_2"):
        encoders.append(("text_encoder_2", pipe.tokenizer_2, pipe.text_encoder_2))

    if not encoders:
        raise RuntimeError("No text encoders found on this pipeline.")

    return pipe, encoders


def filter_tokens(tokenizer, english_vocab, nlp):
    # Apply spaCy POS filtering and English word checks to vocabulary tokens
    tokens = tokenizer.get_vocab()
    id_to_token = {i: t for t, i in tokens.items()}
    all_tokens = [id_to_token[i] for i in range(len(id_to_token))]

    docs = list(nlp.pipe(all_tokens, batch_size=512))
    valid_token_ids = []

    for i, doc in enumerate(docs):
        if (
            len(doc) == 1 and
            doc[0].pos_ in POS_TAGS and
            len(all_tokens[i]) >= MIN_TOKEN_LENGTH and
            all_tokens[i].isalpha() and
            all_tokens[i].lower() == all_tokens[i] and
            all_tokens[i].lower() in english_vocab
        ):
            valid_token_ids.append(i)

    filtered_tokens = [all_tokens[i] for i in valid_token_ids]
    return valid_token_ids, filtered_tokens


def main():
    parser = argparse.ArgumentParser(description="Find high-variance embedding dimensions for SD 1.x or SDXL.")
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--top-n-dimensions", type=int, default=20)
    parser.add_argument("--top-k-tokens", type=int, default=10)
    parser.add_argument("--encoder", choices=["auto", "text_encoder", "text_encoder_2"], default="auto")
    args = parser.parse_args()

    english_vocab = set(w.lower() for w in words.words())
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    pipe, encoders = load_pipelines(args.model_id)

    for enc_name, tokenizer, text_encoder in encoders:
        if args.encoder != "auto" and enc_name != args.encoder:
            continue

        embeddings = text_encoder.get_input_embeddings().weight.data.cpu().numpy()
        valid_ids, filtered_tokens = filter_tokens(tokenizer, english_vocab, nlp)

        if not valid_ids:
            print(f"No valid tokens after filtering for encoder {enc_name}.")
            continue

        filtered_embeddings = embeddings[valid_ids]
        stds = np.std(filtered_embeddings, axis=0)
        top_dims = np.argsort(stds)[-args.top_n_dimensions:][::-1]

        print(f"\n📊 Encoder {enc_name}: analyzing {len(filtered_tokens)} tokens, dim={embeddings.shape[1]}")

        for rank, dim in enumerate(top_dims, 1):
            dim_values = filtered_embeddings[:, dim]
            top_indices = np.argsort(dim_values)[-args.top_k_tokens:][::-1]
            bottom_indices = np.argsort(dim_values)[:args.top_k_tokens]

            print(f"\n📊 Rank {rank}: Dimension {dim} -> std = {stds[dim]:.5f}")
            print("  🔼 Top tokens:")
            for i in top_indices:
                print(f"    {filtered_tokens[i]:<16} -> {dim_values[i]:.4f}")
            print("  🔽 Bottom tokens:")
            for i in bottom_indices:
                print(f"    {filtered_tokens[i]:<16} -> {dim_values[i]:.4f}")
            print("-" * 50)


if __name__ == "__main__":
    main()
