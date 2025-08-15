import argparse

import torch
from diffusers import StableDiffusionPipeline


def load_pipelines(model_id: str):
    # Load the Stable Diffusion pipeline and extract available text encoders
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


def analyze_dimension(encoder_name, tokenizer, text_encoder, target_dim, top_k):
    # Inspect one embedding dimension and print the top/bottom scoring tokens
    text_encoder = text_encoder.eval().float()
    embedding_matrix = text_encoder.get_input_embeddings().weight.data  # shape: [vocab, dim]

    if target_dim < 0 or target_dim >= embedding_matrix.shape[1]:
        raise ValueError(
            f"target_dim {target_dim} out of range for encoder {encoder_name} "
            f"with dim {embedding_matrix.shape[1]}"
        )

    values = embedding_matrix[:, target_dim]
    sorted_indices = torch.argsort(values, descending=True)

    print(f"\n📊 Encoder {encoder_name} - Dimension {target_dim} of {embedding_matrix.shape[1]}:")

    print("\nTop activating tokens:")
    for idx in sorted_indices[:top_k]:
        token = tokenizer.convert_ids_to_tokens(int(idx))
        print(f"{token:<20} -> {values[idx].item():.5f}")

    print("\nLowest activating tokens:")
    for idx in sorted_indices[-top_k:]:
        token = tokenizer.convert_ids_to_tokens(int(idx))
        print(f"{token:<20} -> {values[idx].item():.5f}")


def main():
    # Parse CLI arguments and run analysis
    parser = argparse.ArgumentParser(
        description="Analyze a single embedding dimension for SD 1.x or SDXL models."
    )
    parser.add_argument(
        "--model-id",
        default="runwayml/stable-diffusion-v1-5",
        help="Diffusers model id"
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=1,
        help="Embedding dimension index to inspect"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many tokens to list at each end"
    )
    parser.add_argument(
        "--encoder",
        choices=["auto", "text_encoder", "text_encoder_2"],
        default="auto",
        help="Which encoder to analyze"
    )
    args = parser.parse_args()

    pipe, encoders = load_pipelines(args.model_id)

    for enc_name, tok, enc in encoders:
        if args.encoder != "auto" and enc_name != args.encoder:
            continue
        analyze_dimension(enc_name, tok, enc, args.dimension, args.top_k)


if __name__ == "__main__":
    main()
