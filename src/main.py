import os
import argparse
import logging
from utils import setup_logger, create_dir, save_json, load_yaml
from pipeline import (
    segment_text,
    generate_embeddings,
    classify_local,
    cluster_sentences,
    classify_global,
    merge_local_global,
    final_classification
)

def main():
    # Initialize logger
    logger = setup_logger(log_file="logs/pipeline.log")
    logger.info("Starting Interview Analyzer Ensemble Pipeline")

    # Load configuration from config.yaml
    config_path = "config.yaml"
    config = load_yaml(config_path)
    logger.info(f"Loaded configuration from {config_path}")

    # Ensure necessary directories exist
    create_dir(config["paths"]["logs_dir"])
    create_dir(config["paths"]["output_dir"])

    # Parse command-line arguments for input file
    parser = argparse.ArgumentParser(description="Interview Analyzer Ensemble Pipeline")
    parser.add_argument("--input_file", type=str, default="data/interviews/sample_interview.txt",
                        help="Path to the input transcript file")
    args = parser.parse_args()
    input_file = args.input_file
    logger.info(f"Processing input file: {input_file}")

    # Read the transcript file
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Step 1: Sentence Segmentation
    sentences = segment_text(text)
    logger.info(f"Segmented text into {len(sentences)} sentences.")

    # Step 2: Generate embeddings with context
    embeddings = generate_embeddings(sentences, context_window=config["preprocessing"]["context_window"])
    logger.info("Generated embeddings for all sentences.")

    # Step 3: Local Classification using LLaMA-2 (via prompt engineering)
    df_local = classify_local(sentences, embeddings, config)
    logger.info("Completed local classification.")

    # Step 4: Global Thematic Clustering and Classification
    cluster_labels = cluster_sentences(embeddings, config)
    df_global = classify_global(sentences, embeddings, cluster_labels, config)
    logger.info("Completed global classification.")

    # Step 5: Merge local and global outputs with conflict resolution
    merged_df = merge_local_global(df_local, df_global, config)
    logger.info("Merged local and global outputs.")

    # Step 6: Final Meta-Classification
    final_df = final_classification(merged_df, config)
    logger.info("Final classification complete.")

    # Save final output as JSON in the output directory
    output_path = os.path.join(config["paths"]["output_dir"], "final_output.json")
    final_output = final_df.to_dict(orient="records")
    save_json(final_output, output_path)
    logger.info(f"Final output saved to {output_path}")

if __name__ == "__main__":
    main()
