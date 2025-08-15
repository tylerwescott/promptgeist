import argparse

import torch
from torch.nn.functional import cosine_similarity
from diffusers import StableDiffusionPipeline

import nltk
from nltk.corpus import words

# Ensure the NLTK English word list is available
try:
    nltk.data.find("corpora/words")
except LookupError:
    nltk.download("words")


def load_pipelines(model_id: str):
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


def is_single_token(tokenizer, word):
    tokens = tokenizer.tokenize(word)
    return len(tokens) == 1


def get_related_tokens(tokenizer, embedding_weights, target_emb, top_k=10):
    if torch.isnan(target_emb).any() or torch.isinf(target_emb).any():
        return []

    sims = cosine_similarity(target_emb.unsqueeze(0), embedding_weights)

    try:
        top_ids = sims.topk(top_k + 200).indices.tolist()
    except RuntimeError:
        return []

    id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
    related = []
    seen = set()

    for tid in top_ids:
        token = id_to_token.get(tid, "")
        if not token or token.startswith("<") or token.startswith("[") or "###" in token:
            continue

        token_clean = token.replace("</w>", "").strip()
        if not token_clean.isalpha() or token_clean in seen or len(token_clean) < 2:
            continue

        token_input_ids = tokenizer(token_clean, return_tensors="pt")["input_ids"][0]
        if len(token_input_ids) != 3:
            continue

        token_emb = embedding_weights[token_input_ids[1]]
        norm = token_emb.norm().item()
        sim = cosine_similarity(token_emb.unsqueeze(0), target_emb.unsqueeze(0)).item()

        if sim > 0.1 and norm > 0.30:
            related.append((token_clean, sim))
            seen.add(token_clean)

        if len(related) >= top_k:
            break

    return related


def inspect_token(tokenizer, embedding_weights, word, reference_words):
    input_ids = tokenizer(word, return_tensors="pt")["input_ids"][0]
    tokenized = tokenizer.tokenize(word)

    if len(tokenized) != 1:
        return word, "fragmented", None, None, []

    token_id = input_ids[1]
    token_emb = embedding_weights[token_id]
    norm = token_emb.norm().item()

    ref_ids = [
        tokenizer(w, return_tensors="pt")["input_ids"][0][1]
        for w in reference_words if is_single_token(tokenizer, w)
    ]
    ref_embs = [embedding_weights[i] for i in ref_ids]

    if not ref_embs:
        avg_sim = 0.0
    else:
        sims = [cosine_similarity(token_emb.unsqueeze(0), ref.unsqueeze(0)).item() for ref in ref_embs]
        avg_sim = sum(sims) / len(sims)

    if avg_sim > 0.3 and norm > 0.35:
        status = "recognized"
    elif avg_sim > 0.1 or norm > 0.30:
        status = "weak"
    else:
        status = "unknown"

    related = get_related_tokens(tokenizer, embedding_weights, token_emb)
    return word, status, norm, avg_sim, related


def main():
    parser = argparse.ArgumentParser(description="Token recognition and similarity inspection for SD 1.x or SDXL.")
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--encoder", choices=["auto", "text_encoder", "text_encoder_2"], default="auto")
    parser.add_argument("--words", nargs="*", default=[
        "ecuador", "grab", "floor", "wordpress", "lava", "massive", "visional",
        "tapered", "signation", "mexico", "asdfghjkl", "dragon", "cat", "xyzabc",
        "twilight", "margo", "bombard"
    ])
    args = parser.parse_args()

    pipe, encoders = load_pipelines(args.model_id)

    for enc_name, tokenizer, text_encoder in encoders:
        if args.encoder != "auto" and enc_name != args.encoder:
            continue

        embedding_weights = text_encoder.get_input_embeddings().weight
        print(f"\n===== Encoder: {enc_name} | dim={embedding_weights.shape[1]} =====")
        reference_words = ["cat", "city", "sky"]

        print(f"{'Token':<12} {'Status':<12} {'Norm':<10} {'Similarity':<10} {'RelatedTo'}")
        print("-" * 70)

        for word in args.words:
            word, status, norm, sim, related = inspect_token(tokenizer, embedding_weights, word, reference_words)
            if status == "fragmented":
                print(f"{word:<12} fragmented   {'-':<10} {'-':<10} -")
            else:
                related_str = ", ".join([f"{tok}({score:.2f})" for tok, score in related])
                print(f"{word:<12} {status.upper():<12} {norm:.4f}     {sim:.4f}     {related_str}")

        # Scan English words for lowest embedding norm
        print("\nScanning for real English words with lowest embedding norm...\n")
        english_words = set(w.lower() for w in words.words())
        word_norms = []

        for w in english_words:
            tokenized = tokenizer.tokenize(w)
            if len(tokenized) != 1:
                continue

            input_ids = tokenizer(w, return_tensors="pt")["input_ids"][0]
            if len(input_ids) != 3:
                continue

            token_id = input_ids[1]
            token_emb = embedding_weights[token_id]
            norm = token_emb.norm().item()
            word_norms.append((w, norm))

        word_norms.sort(key=lambda x: x[1])
        print(f"{'Word':<15} {'Norm':<10}")
        print("-" * 30)
        for w, norm in word_norms[:100]:
            print(f"{w:<15} {norm:.4f}")
        print(f"\nTotal single-token English words checked: {len(word_norms)}")

        # Scan all tokens for highest embedding norm
        print("\nScanning for highest-norm tokens in the vocabulary...\n")
        id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
        high_norm_tokens = []

        for token_id, token in id_to_token.items():
            token_clean = token.replace("</w>", "").strip()
            if not token_clean.isalpha() or len(token_clean) < 2:
                continue

            token_input_ids = tokenizer(token_clean, return_tensors="pt")["input_ids"][0]
            if len(token_input_ids) != 3:
                continue

            token_emb = embedding_weights[token_input_ids[1]]
            norm = token_emb.norm().item()
            high_norm_tokens.append((token_clean, norm))

        high_norm_tokens.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Token':<20} {'Norm':<10}")
        print("-" * 32)
        for token, norm in high_norm_tokens[:100]:
            print(f"{token:<20} {norm:.4f}")
        print(f"\nTotal scanned tokens: {len(high_norm_tokens)}")


if __name__ == "__main__":
    main()
