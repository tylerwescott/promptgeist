import argparse

import numpy as np
import torch
from diffusers import StableDiffusionPipeline


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


def analyze_token(token_str, tokenizer, embeddings, top_n=10):
    # Show top/bottom activated dimensions for a given token
    vocab = tokenizer.get_vocab()
    if token_str not in vocab:
        print(f"❌ Token '{token_str}' not found in this tokenizer's vocabulary.")
        return

    token_id = vocab[token_str]
    embedding_vector = embeddings[token_id]

    top_dims = np.argsort(embedding_vector)[-top_n:][::-1]
    bottom_dims = np.argsort(embedding_vector)[:top_n]

    print(f"\n📊 Token '{token_str}' (ID {token_id})")
    print(f"Norm: {np.linalg.norm(embedding_vector):.4f}")

    print("\n🔼 Top activated dimensions:")
    for dim in top_dims:
        print(f"  Dimension {dim:<4} -> {embedding_vector[dim]:+.5f}")

    print("\n🔽 Bottom activated dimensions:")
    for dim in bottom_dims:
        print(f"  Dimension {dim:<4} -> {embedding_vector[dim]:+.5f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze one token's embedding dimensions for SD 1.x or SDXL.")
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5", help="Diffusers model id")
    parser.add_argument("--token", required=False, help="Exact tokenizer token string, like 'cat' or 'Ġcat' depending on vocab")
    parser.add_argument("--top-n", type=int, default=10, help="How many dims to show for top and bottom")
    parser.add_argument("--encoder", choices=["auto", "text_encoder", "text_encoder_2"], default="auto", help="Which encoder to analyze")
    args = parser.parse_args()

    pipe, encoders = load_pipelines(args.model_id)

    if args.token is None:
        print("Enter tokens to analyze. Type 'exit' to quit.")

    for enc_name, tok, enc in encoders:
        if args.encoder != "auto" and enc_name != args.encoder:
            continue

        embeddings = enc.get_input_embeddings().weight.data.cpu().numpy()
        vocab = tok.get_vocab()
        print(f"\n===== Encoder: {enc_name} | dim={embeddings.shape[1]} | vocab={len(vocab)} =====")

        if args.token:
            analyze_token(args.token, tok, embeddings, args.top_n)
        else:
            while True:
                token_input = input("\nToken: ").strip()
                if token_input.lower() == "exit":
                    break
                analyze_token(token_input, tok, embeddings, args.top_n)


if __name__ == "__main__":
    main()
