import pandas as pd
import torch
from tqdm import tqdm
import argparse
import pickle
from transformers import T5Tokenizer, T5Model


def save_pickle(data, filename):
    """Save data to a pickle file."""
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def main(args):
    print("Loading data...")
    items_df = pd.read_csv(args.input_csv)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5Model.from_pretrained(args.model_name).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("Computing embeddings...")
    embeddings = {}
    batch_size = args.batch_size

    for start_idx in tqdm(
        range(0, items_df.shape[0], batch_size), desc="Embedding Computation"
    ):
        batch = items_df.iloc[start_idx : start_idx + batch_size]
        batch_texts = [
            " ".join([str(row["title"]), str(row["description"])]).strip()
            for _, row in batch.iterrows()
        ]
        batch_ids = [str(row["item_id"]) for _, row in batch.iterrows()]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.encoder(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        for item_id, embedding in zip(batch_ids, batch_embeddings):
            embeddings[item_id] = embedding

    print("Saving final embeddings...")
    save_pickle(embeddings, f"{args.output_prefix}_embeddings.pkl")

    print("All files have been saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute embeddings for items using T5-base and save them as pickle."
    )
    parser.add_argument(
        "--input_csv", type=str, required=True, help="Path to the input CSV file."
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        required=True,
        help="Prefix for the output files.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="t5-base",
        help="Name of the T5 model to use for embedding computation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding computation.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Number of batches to process before saving intermediate results.",
    )
    args = parser.parse_args()
    main(args)
