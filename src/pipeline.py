"""
pipeline.py
-----------
This module contains the main pipeline functions for processing interview transcripts.
It covers sentence segmentation, embedding generation with context, local classification,
thematic clustering and global classification, merging, conflict resolution, and final classification.
"""

import re
import numpy as np
import pandas as pd
import spacy
import logging
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.preprocessing import StandardScaler

# Load spaCy model (English)
nlp = spacy.load("en_core_web_sm")

# Global logging instance (assumes logging is already configured)
logger = logging.getLogger(__name__)

# Initialize Sentence Transformer model for embedding generation.
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)

# Import the LLaMA-2 model loader from ctransformers.
try:
    from ctransformers import AutoModelForCausalLM
except ImportError:
    raise ImportError("ctransformers library not installed. Please install it to use LLaMA-2.")

############################################
# 1. Sentence Segmentation
############################################
def segment_text(text):
    """
    Splits unstructured text into sentences using spaCy.
    
    Args:
        text (str): The full unstructured transcript text.
    
    Returns:
        list of dict: Each dict contains 'id' and 'sentence' keys.
    """
    logger.info("Segmenting text into sentences using spaCy.")
    doc = nlp(text)
    sentences = []
    for i, sent in enumerate(doc.sents):
        sentence_text = sent.text.strip()
        if sentence_text:
            sentences.append({"id": i, "sentence": sentence_text})
    logger.info(f"Segmented text into {len(sentences)} sentences.")
    return sentences

############################################
# 2. Embedding Generation with Contextual Enrichment
############################################
def generate_embeddings(sentences, context_window=1):
    """
    Generate embeddings for each sentence and enrich them with a context window.
    
    Args:
        sentences (list of dict): List with each sentence and its ID.
        context_window (int): Number of neighboring sentences to include on each side.
    
    Returns:
        np.array: Array of enriched embeddings.
    """
    logger.info("Generating base embeddings using Sentence Transformers.")
    sentence_texts = [s["sentence"] for s in sentences]
    base_embeddings = embedder.encode(sentence_texts, convert_to_numpy=True)
    
    enriched_embeddings = []
    total = len(base_embeddings)
    for i in range(total):
        # Define indices for the context window.
        start = max(0, i - context_window)
        end = min(total, i + context_window + 1)
        # Compute the average embedding over the window.
        context_embedding = np.mean(base_embeddings[start:end], axis=0)
        enriched_embeddings.append(context_embedding)
    
    logger.info("Generated context-aware embeddings for all sentences.")
    return np.array(enriched_embeddings)

############################################
# Initialize the LLaMA-2 model
############################################
# Using ctransformers version 0.2.27 as installed, we initialize the model.
llama_model = AutoModelForCausalLM.from_pretrained("/models/llama-2-7b-chat.Q4_K_M.gguf")
llama_model.max_new_tokens = 30

############################################
# 3. Local Classification (Sentence-Level)
############################################
def classify_local(sentences, embeddings, config):
    """
    Classify each sentence individually using LLaMA-2 via prompt engineering.
    This function builds a prompt from the configured template, sends it to the model,
    and parses the response to extract the topic label.

    Args:
        sentences (list of dict): List of sentence dictionaries with keys 'id' and 'sentence'.
        embeddings (np.array): Array of enriched embeddings (not used directly here).
        config (dict): Configuration parameters containing the local prompt and thresholds.

    Returns:
        pd.DataFrame: DataFrame with columns: id, sentence, local_label, local_confidence.
    """
    logger.info("Performing local classification on each sentence using LLaMA-2.")
    results = []
    local_prompt = config["classification"]["local"]["prompt"]
    confidence_threshold = config["classification"]["local"]["confidence_threshold"]

    for item in sentences:
        # For now, we use an empty context; extend this if needed.
        full_prompt = (
            f"{local_prompt}\n\n"
            f"Input Sentence: \"{item['sentence']}\"\n"
            f"Context: []"
        )
        
        # Call the model using the generate method as documented.
        response = llama_model(full_prompt, max_new_tokens=30)
        logger.debug(f"Model response for sentence ID {item['id']}: {response}")

        # Parse the response to extract the label (assumed to be enclosed in angle brackets).
        match = re.search(r"<(.*?)>", response)
        label = match.group(1).strip() if match else "Unknown"
        confidence = 0.9 if label != "Unknown" else 0.5

        results.append({
            "id": item["id"],
            "sentence": item["sentence"],
            "local_label": label,
            "local_confidence": confidence
        })

    df_local = pd.DataFrame(results)
    logger.info("Local classification completed for all sentences.")
    return df_local

############################################
# 4. Global Thematic Clustering & Classification
############################################
def cluster_sentences(embeddings, config):
    """
    Cluster sentence embeddings using HDBSCAN.
    
    Args:
        embeddings (np.array): Array of enriched sentence embeddings.
        config (dict): Configuration parameters for clustering.
    
    Returns:
        np.array: Array of cluster labels (with -1 for outliers).
    """
    logger.info("Clustering sentence embeddings using HDBSCAN.")
    hdbscan_params = config["clustering"]["hdbscan"]
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_params.get("min_cluster_size", 5),
        metric=hdbscan_params.get("metric", "euclidean")
    )
    cluster_labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    logger.info(f"HDBSCAN produced {n_clusters} clusters (excluding outliers).")
    return cluster_labels

def classify_global(sentences, embeddings, cluster_labels, config):
    """
    Perform global thematic classification by aggregating embeddings for each cluster.
    For each cluster (except outliers), this function concatenates a few representative sentences
    to form a cluster summary, builds a prompt using the global prompt template, and sends it to LLaMA-2.
    The response is parsed to extract a thematic label.

    Args:
        sentences (list of dict): List of sentence dictionaries.
        embeddings (np.array): Array of enriched embeddings.
        cluster_labels (np.array): Cluster labels for each sentence.
        config (dict): Configuration with global classification parameters.
    
    Returns:
        pd.DataFrame: DataFrame with columns: id, global_label, cluster.
    """
    logger.info("Performing global thematic classification on clusters.")
    df = pd.DataFrame({
        "id": [s["id"] for s in sentences],
        "sentence": [s["sentence"] for s in sentences],
        "cluster": cluster_labels
    })
    
    global_results = []
    global_prompt = config["classification"]["global"]["prompt"]
    
    for cluster in sorted(df["cluster"].unique()):
        if cluster == -1:
            global_label = "Unassigned"
            cluster_ids = df[df["cluster"] == cluster]["id"].tolist()
        else:
            cluster_sentences = df[df["cluster"] == cluster]["sentence"].tolist()
            cluster_summary = " ".join(cluster_sentences[:3])
            full_prompt = (
                f"{global_prompt}\n\n"
                f"Cluster Summary: \"{cluster_summary}\""
            )
            
            response = llama_model(full_prompt, max_new_tokens=30)
            logger.debug(f"Model response for cluster {cluster}: {response}")
            
            match = re.search(r"<(.*?)>", response)
            global_label = match.group(1).strip() if match else "Unknown"
            cluster_ids = df[df["cluster"] == cluster]["id"].tolist()
        
        global_results.append({
            "cluster": cluster,
            "global_label": global_label,
            "sentence_ids": cluster_ids
        })
    
    global_label_map = {}
    for entry in global_results:
        for sid in entry["sentence_ids"]:
            global_label_map[sid] = entry["global_label"]
    
    df["global_label"] = df["id"].apply(lambda x: global_label_map.get(x, "Unassigned"))
    logger.info("Global thematic classification completed.")
    return df[["id", "global_label", "cluster"]]

############################################
# 5. Merging Local & Global Outputs and Conflict Resolution
############################################
def merge_local_global(df_local, df_global, config):
    """
    Merge local and global classification results on sentence ID.
    
    Args:
        df_local (pd.DataFrame): DataFrame with local classification results.
        df_global (pd.DataFrame): DataFrame with global thematic classification.
        config (dict): Configuration for conflict resolution.
    
    Returns:
        pd.DataFrame: Merged DataFrame including a preliminary final label.
    """
    logger.info("Merging local and global outputs.")
    merged_df = pd.merge(df_local, df_global, on="id", how="left")
    merged_df["final_label"] = merged_df.apply(
        lambda row: resolve_conflict(
            row["local_label"],
            row["global_label"],
            row["local_confidence"],
            config
        ),
        axis=1
    )
    logger.info("Merging and conflict resolution completed.")
    return merged_df

def resolve_conflict(local_label, global_label, local_confidence, config):
    """
    Resolve conflicts between local and global labels using weighted rules.
    
    Args:
        local_label (str): Label predicted at the local level.
        global_label (str): Label predicted at the global (cluster) level.
        local_confidence (float): Confidence score from local classification.
        config (dict): Configuration parameters including thresholds and weights.
    
    Returns:
        str: The final chosen label.
    """
    threshold = config["classification"]["local"].get("confidence_threshold", 0.8)
    weight_local = config["classification"]["final"].get("final_weight_local", 0.6)
    weight_global = config["classification"]["final"].get("final_weight_global", 0.4)
    
    if local_confidence >= threshold:
        return local_label
    else:
        return global_label

############################################
# 6. Final Meta-Classification / Post-Processing
############################################
def final_classification(merged_df, config):
    """
    Perform final meta-classification. This function is a placeholder for
    additional feature engineering or meta-classifier training.
    
    Args:
        merged_df (pd.DataFrame): DataFrame after merging local and global outputs.
        config (dict): Configuration parameters.
    
    Returns:
        pd.DataFrame: Final DataFrame with classification results.
    """
    logger.info("Performing final meta-classification.")
    scaler = StandardScaler()
    merged_df["local_conf_norm"] = scaler.fit_transform(merged_df[["local_confidence"]])
    logger.info("Final classification complete.")
    return merged_df
