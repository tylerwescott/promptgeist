import argparse

import torch
from diffusers import StableDiffusionPipeline
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt


def load_pipelines(model_id: str):
    # Load SD pipeline and extract available text encoders
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


def gather_embeddings(token_list, tokenizer, text_encoder):
    # Extract embedding vectors for a list of valid single-token inputs
    embedding_weights = text_encoder.get_input_embeddings().weight
    valid_tokens = []
    embeddings = []

    for token in token_list:
        tokenized = tokenizer.tokenize(token)
        if len(tokenized) != 1:
            continue  # skip multi-token inputs

        input_ids = tokenizer(token, return_tensors="pt")["input_ids"][0]
        if len(input_ids) != 3:
            continue  # expect [BOS, token, EOS]

        token_id = input_ids[1]
        emb = embedding_weights[token_id].detach().cpu().numpy()
        valid_tokens.append(token)
        embeddings.append(emb)

    return valid_tokens, embeddings


def plot_2d(points, labels, title):
    # 2D scatterplot with token labels
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=14)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    plt.scatter(xs, ys)
    for i, tok in enumerate(labels):
        plt.annotate(tok, (xs[i], ys[i]), fontsize=9)

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize token embeddings with PCA and UMAP for SD 1.x or SDXL.")
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--encoder", choices=["auto", "text_encoder", "text_encoder_2"], default="auto")
    parser.add_argument("--tokens", nargs="*", default=[
        "cat", "dog", "tiger", "lion", "cheetah", "robot", "android", "cyborg",
        "car", "truck", "pizza", "burger", "salad", "sushi", "dragon", "unicorn",
        "goblin", "wizard", "city", "castle"
    ])
    args = parser.parse_args()

    pipe, encoders = load_pipelines(args.model_id)

    for enc_name, tok, enc in encoders:
        if args.encoder != "auto" and enc_name != args.encoder:
            continue

        valid_tokens, embs = gather_embeddings(args.tokens, tok, enc)
        if len(embs) < 3:
            print(f"Not enough single-token items for encoder {enc_name}.")
            continue

        print(f"Visualizing {len(valid_tokens)} tokens for {enc_name}...")

        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(embs)
        plot_2d(pca_coords, valid_tokens, f"PCA Projection - {enc_name}")

        reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, metric="cosine", random_state=42)
        umap_coords = reducer.fit_transform(embs)
        plot_2d(umap_coords, valid_tokens, f"UMAP Projection - {enc_name}")


if __name__ == "__main__":
    main()
