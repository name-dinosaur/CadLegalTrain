import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import torch
import time
import gc
import pickle
import re

MODEL_NAME = 'all-MiniLM-L6-v2'
TOP_N = 10
MAX_TEXT_WORDS = 1500

NUM_CASES_TO_PROCESS = 10
NUM_LAWS_TO_PROCESS = None

# --- Save Paths ---
MODEL_SAVE_PATH = "saved_model"
EMBEDDINGS_SAVE_PATH = "law_embeddings.pkl"
IDS_SAVE_PATH = "law_ids.pkl"


def clean_text_between_citation_and_signature(text):
    citation_pattern = r'(\d{4}\s+CHRT\s+\d{1,2})\s*\n(\d{4}/\d{2}/\d{2})'
    signature_pattern = r'Signed by\s+[A-Za-z\s\-éçàèùâêîôûëïüÿ]+(?:")?\s*\n[A-Za-z\s]+,\s+[A-Za-z\s]+\s*\n(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}'

    citation_match = re.search(citation_pattern, text, re.MULTILINE)
    signature_match = re.search(signature_pattern, text, re.MULTILINE)

    if citation_match and signature_match:
        start_pos = citation_match.start()
        end_pos = signature_match.end()
        cleaned_text = text[start_pos:end_pos]
    elif citation_match:
        start_pos = citation_match.start()
        cleaned_text = text[start_pos:]
    elif signature_match:
        end_pos = signature_match.end()
        cleaned_text = text[:end_pos]
    else:
        cleaned_text = text

    return cleaned_text


def prepare_and_filter_data(dataset, text_field='unofficial_text', id_field='name'):
    """Filters dataset for valid text and ID, cleans, and truncates text."""
    valid_items = []
    for item in dataset:
        text = item.get(text_field)
        identifier = item.get(id_field)
        if text and identifier:
            cleaned_text = clean_text_between_citation_and_signature(text)
            words = cleaned_text.split()
            truncated_text = " ".join(words[:MAX_TEXT_WORDS])
            valid_items.append({'id': identifier, 'text': truncated_text})
    return valid_items


# Loading the specific data instance of 'legislation', 'regulations', 'SCC' from the bulk data of 'https://huggingface.co/datasets/refugee-law-lab/canadian-legal-data'.
if __name__ == '__main__':
    print("Loading datasets...")
    legislation_ds = load_dataset("refugee-law-lab/canadian-legal-data", "LEGISLATION-FED", split='train')
    regulation_ds = load_dataset("refugee-law-lab/canadian-legal-data", "REGULATIONS-FED", split='train')
    supreme_court_cases = load_dataset("refugee-law-lab/canadian-legal-data", "SCC", split='train')
    print("Datasets loaded.")

    print("Preparing and filtering law/regulation data...")
    filtered_legislation = prepare_and_filter_data(legislation_ds)
    filtered_regulations = prepare_and_filter_data(regulation_ds)
    all_laws_data = filtered_legislation + filtered_regulations

    if NUM_LAWS_TO_PROCESS is not None and len(all_laws_data) > NUM_LAWS_TO_PROCESS:
        print(f"Limiting laws/regulations to first {NUM_LAWS_TO_PROCESS} for processing.")
        all_laws_data = all_laws_data[:NUM_LAWS_TO_PROCESS]

    if not all_laws_data:
        print("Error: No valid law/regulation data with text found after filtering. Exiting.")
        exit()

    print(f"Prepared {len(all_laws_data)} laws/regulations for embedding.")
    law_texts = [item['text'] for item in all_laws_data]
    law_ids = [item['id'] for item in all_laws_data]

    del legislation_ds, regulation_ds, filtered_legislation, filtered_regulations
    gc.collect()

    print(f"Loading sentence transformer model: {MODEL_NAME}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    print("Model loaded.")

    

    # Check for saved embeddings
    if os.path.exists(EMBEDDINGS_SAVE_PATH) and os.path.exists(IDS_SAVE_PATH):
        print(f"Loading saved embeddings from {EMBEDDINGS_SAVE_PATH}...")
        law_embeddings = torch.load(EMBEDDINGS_SAVE_PATH, map_location=device)
        with open(IDS_SAVE_PATH, 'rb') as f:
            saved_law_ids = pickle.load(f)

        # Verify loaded IDs match current data
        if saved_law_ids == law_ids:
            print("Loaded embeddings successfully.")
        else:
            print("Warning: Saved law IDs do not match current data. Recomputing embeddings...")
            del law_embeddings, saved_law_ids
            gc.collect()
            law_embeddings = None
    else:
        law_embeddings = None

    # Compute embeddings if not loaded
    if law_embeddings is None:
        print("Computing embeddings for laws/regulations...")
        start_time = time.time()
        law_embeddings = model.encode(law_texts, convert_to_tensor=True, show_progress_bar=True, device=device)
        end_time = time.time()
        print(f"Computed {len(law_embeddings)} embeddings in {end_time - start_time:.2f} seconds.")

        print(f"Saving embeddings to {EMBEDDINGS_SAVE_PATH}...")
        torch.save(law_embeddings, EMBEDDINGS_SAVE_PATH)
        print("Embeddings saved.")

        print(f"Saving law IDs to {IDS_SAVE_PATH}...")
        with open(IDS_SAVE_PATH, 'wb') as f:
            pickle.dump(law_ids, f)
        print("Law IDs saved.")

    del law_texts
    gc.collect()

    results = {}
    processed_count = 0
    start_time_cases = time.time()

    indices_to_process = range(len(supreme_court_cases))
    if NUM_CASES_TO_PROCESS is not None:
        indices_to_process = range(min(NUM_CASES_TO_PROCESS, len(supreme_court_cases)))

    print(f"\nProcessing {len(indices_to_process)} SCC cases to find top {TOP_N} similar laws...")

    for i in indices_to_process:
        case = supreme_court_cases[i]
        case_identifier = case.get('citation') or case.get('name')
        case_text = case.get('unofficial_text')
        print(f"Case identifier:{case_identifier}")
        if not case_identifier or not case_text:
            print(f"Skipping case index {i} (ID: {case_identifier or 'Unknown'}) due to missing identifier or text.")
            continue

        cleaned_case_text = clean_text_between_citation_and_signature(case_text)
        words = cleaned_case_text.split()
        truncated_case_text = " ".join(words[:MAX_TEXT_WORDS])

        if not truncated_case_text:
            print(f"Skipping case {case_identifier} due to empty text after truncation.")
            continue

        case_embedding = model.encode(truncated_case_text, convert_to_tensor=True, device=device)
        if len(case_embedding.shape) == 1:
            case_embedding = case_embedding.unsqueeze(0)

        cosine_scores = util.cos_sim(case_embedding, law_embeddings)[0].cpu()
        top_results = torch.topk(cosine_scores, k=min(TOP_N, len(all_laws_data)))

        top_laws_for_case = []
        indices = top_results.indices.tolist()
        scores = top_results.values.tolist()
        for idx, score in zip(indices, scores):
            top_laws_for_case.append((law_ids[idx], score))

        results[case_identifier] = top_laws_for_case

        processed_count += 1
        if processed_count % 5 == 0:
            elapsed = time.time() - start_time_cases
            print(f"  Processed {processed_count}/{len(indices_to_process)} cases... ({elapsed:.2f} seconds)")

        del case_embedding, cosine_scores, top_results
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    end_time_cases = time.time()
    print(f"Finished processing cases in {end_time_cases - start_time_cases:.2f} seconds.")

    print(f"\n--- Top {TOP_N} Relevant Laws/Regulations (Ranked by Semantic Similarity) ---")
    display_count = 0
    max_display = 10
    for case_id, ranked_laws in results.items():
        if display_count >= max_display:
            print(f"\n[... Trimmed results, showing first {max_display} cases processed ...]")
            break
        print(f"\nCase: {case_id}")
        if not ranked_laws:
            print("  No relevant laws found (or processed).")
        else:
            for i, (law_name, score) in enumerate(ranked_laws):
                print(f"  {i + 1}. {law_name} (Similarity: {score:.4f})")
        display_count += 1
