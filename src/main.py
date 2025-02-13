import os
import json
import logging
import pandas as pd
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.preprocessing import StandardScaler

# Initialize logging (this is a simple logging strategy)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler("pipeline.log")
                    ])
logger = logging.getLogger(__name__)

# Global configuration (can later be loaded from a YAML/JSON config file)
CONFIG = {
    "context_window": 1,  # number of previous/next sentences to include
    "hdbscan_min_cluster_size": 5,
    "embedding_model_name": "all-MiniLM-L6-v2",
    "local_classification_prompt": "Your LLaMA-2 prompt goes here.",
    "global_classification_prompt": "Global classification prompt for cluster summary.",
    "final_weight_local": 0.6,
    "final_weight_global": 0.4
}

# Load spaCy model once (assuming English)
nlp = spacy.load("en_core_web_sm")

# Initialize Sentence Transformer model
embedder = SentenceTransformer(CONFIG["embedding_model_name"])

############################################
# 1. Bottom-Up: Sentence Segmentation
############################################
def segment_text(text):
    """
    Splits unstructured text into sentences using spaCy and assigns each a unique ID.
    Returns a list of dictionaries: [{"id": 0, "sentence": "..."}, ...].
    """
    logger.info("Segmenting text into sentences.")
    doc = nlp(text)
    sentences = []
    for i, sent in enumerate(doc.sents):
        sentence_text = sent.text.strip()
        if sentence_text:
            sentences.append({"id": i, "sentence": sentence_text})
    logger.info(f"Segmented text into {len(sentences)} sentences.")
    return sentences

############################################
# 2. Embedding Generation & Contextual Enrichment
############################################
def generate_embeddings(sentences, context_window=CONFIG["context_window"]):
    """
    Generate embeddings for each sentence using Sentence Transformers.
    Optionally enrich each sentence embedding with a context window from neighbors.
    Returns a list of embeddings (NumPy arrays).
    """
    logger.info("Generating base embeddings for each sentence.")
    sentence_texts = [s["sentence"] for s in sentences]
    base_embeddings = embedder.encode(sentence_texts, convert_to_numpy=True)
    
    # Create context-aware embeddings by combining neighbor embeddings
    enriched_embeddings = []
    total = len(base_embeddings)
    for i in range(total):
        # Define context boundaries
        start = max(0, i - context_window)
        end = min(total, i + context_window + 1)
        # Aggregate embeddings (e.g., by averaging)
        context_embedding = np.mean(base_embeddings[start:end], axis=0)
        enriched_embeddings.append(context_embedding)
    
    logger.info("Generated enriched embeddings for all sentences.")
    return np.array(enriched_embeddings)

############################################
# 3. Local Classification with LLaMA-2
############################################
def classify_local(sentences, embeddings):
    """
    Classify each sentence locally using LLaMA-2 via prompt engineering.
    This is a stub; replace with actual model calls or parameter-efficient tuning logic.
    Returns a DataFrame with columns: id, sentence, local_label, local_confidence.
    """
    logger.info("Performing local classification for each sentence.")
    results = []
    for idx, item in enumerate(sentences):
        # Placeholder for the classification logic using LLaMA-2.
        # Here we simulate with dummy labels and random confidence.
        dummy_label = "Topic_A"  # You would replace this with the model’s output.
        dummy_confidence = np.random.uniform(0.7, 1.0)
        results.append({
            "id": item["id"],
            "sentence": item["sentence"],
            "local_label": dummy_label,
            "local_confidence": dummy_confidence
        })
    df_local = pd.DataFrame(results)
    logger.info("Completed local classification.")
    return df_local

############################################
# 4. Top-Down: Thematic Clustering & Global Classification
############################################
def cluster_sentences(embeddings):
    """
    Use HDBSCAN to cluster sentence embeddings.
    Returns an array of cluster labels (with -1 for noise/outliers).
    """
    logger.info("Clustering sentences using HDBSCAN.")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=CONFIG["hdbscan_min_cluster_size"],
                                metric="euclidean")
    cluster_labels = clusterer.fit_predict(embeddings)
    logger.info(f"Identified {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters.")
    return cluster_labels

def classify_global(sentences, embeddings, cluster_labels):
    """
    Aggregate embeddings per cluster and perform global classification.
    Uses LLaMA-2 prompt engineering on aggregated embeddings.
    Returns a DataFrame mapping sentence IDs to global thematic labels.
    """
    logger.info("Performing global (thematic) classification.")
    df = pd.DataFrame({
        "id": [s["id"] for s in sentences],
        "sentence": [s["sentence"] for s in sentences],
        "cluster": cluster_labels
    })
    
    # Compute cluster-level (global) embeddings as centroids
    global_results = []
    for cluster in sorted(df["cluster"].unique()):
        if cluster == -1:
            # Handle outliers separately; for example, assign a default label.
            global_label = "Unassigned"
            cluster_ids = df[df["cluster"] == cluster]["id"].tolist()
        else:
            indices = df[df["cluster"] == cluster].index.tolist()
            centroid = np.mean(embeddings[indices], axis=0)
            # Placeholder for actual LLaMA-2 prompt classification of the centroid.
            global_label = "Global_Topic_A"  # Replace with actual output.
            cluster_ids = df[df["cluster"] == cluster]["id"].tolist()
        
        global_results.append({
            "cluster": cluster,
            "global_label": global_label,
            "sentence_ids": cluster_ids
        })
    
    # Map global labels back to each sentence.
    global_label_map = {}
    for entry in global_results:
        for sid in entry["sentence_ids"]:
            global_label_map[sid] = entry["global_label"]
    
    df["global_label"] = df["id"].apply(lambda x: global_label_map.get(x, "Unassigned"))
    logger.info("Completed global classification.")
    return df[["id", "global_label", "cluster"]]

############################################
# 5. Merging Local & Global Outputs
############################################
def merge_local_global(df_local, df_global):
    """
    Merge local and global classification results on the sentence ID.
    Returns a merged DataFrame.
    """
    logger.info("Merging local and global outputs.")
    merged_df = pd.merge(df_local, df_global, on="id", how="left")
    # Optional: Resolve conflicts here if labels differ using weighted rules.
    # For now, add a placeholder column for final_label.
    merged_df["final_label"] = merged_df.apply(
        lambda row: resolve_conflict(row["local_label"], row["global_label"],
                                     row["local_confidence"]), axis=1
    )
    logger.info("Completed merging and initial conflict resolution.")
    return merged_df

def resolve_conflict(local_label, global_label, local_confidence):
    """
    Resolve conflict between local and global labels using weighted rules.
    Currently, this is a placeholder that prefers the local label when confidence is high.
    """
    # Example logic: if local_confidence is high, trust local; otherwise, use global.
    threshold = 0.8  # This threshold can be adjusted via configuration.
    if local_confidence >= threshold:
        return local_label
    else:
        return global_label

############################################
# 6. Final Classification / Meta-Classification
############################################
def final_classification(merged_df):
    """
    Optionally, perform further feature engineering or a meta-classifier step.
    For now, we assume the merged_df final_label is the final output.
    You can extend this function to add more sophisticated meta-classification.
    """
    logger.info("Finalizing classification results.")
    # For demonstration, we normalize confidence scores and output final decisions.
    scaler = StandardScaler()
    merged_df["local_conf_norm"] = scaler.fit_transform(merged_df[["local_confidence"]])
    
    # This is where a meta-classifier could be applied. For now, we simply return the merged DataFrame.
    return merged_df

############################################
# 7. Main Function: Orchestrate the Pipeline
############################################
def main(input_file, output_json="final_output.json"):
    # Read the unstructured text file
    logger.info(f"Reading input file: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Step 1: Sentence segmentation
    sentences = segment_text(text)
    
    # Step 2: Generate embeddings (with context)
    embeddings = generate_embeddings(sentences, context_window=CONFIG["context_window"])
    
    # Step 3: Local classification (per sentence)
    df_local = classify_local(sentences, embeddings)
    
    # Step 4: Global (thematic) clustering and classification
    cluster_labels = cluster_sentences(embeddings)
    df_global = classify_global(sentences, embeddings, cluster_labels)
    
    # Step 5: Merge local and global outputs and resolve conflicts
    merged_df = merge_local_global(df_local, df_global)
    
    # Step 6: Final meta-classification (if applicable)
    final_df = final_classification(merged_df)
    
    # Save final results as JSON
    final_output = final_df.to_dict(orient="records")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
    logger.info(f"Final output saved to {output_json}")

if __name__ == "__main__":
    # Example: run the pipeline on a sample transcript file.
    input_filename = "meeting_transcript.txt"
    main(input_filename)
