import os
import json
import sys  # Added to redirect print output
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict

#CONFIG
AGGREGATED_HUMAN_JSON_DIR = "consensus_human_results\json"

MODEL_JSON_DIR = r"D:\tfg\TFG_FINAL\pdocvqa_satml\Results_Occluded_FINAL\model_attention_maps" #r"C:\Users\marta\tfg\pdocvqa_satml\Results_Occluded_FINAL\model_attention_maps"
OCR_DATA_PATH = r"D:\tfg\TFG_FINAL\filtered_imdb_train_landscape.npy" #r"C:/Users/marta/tfg/filtered_imdb_train_landscape.npy"

REPORT_FILENAME = "unified_analysis_report_Occluded.txt"


def find_file_pairs(human_dir, model_dir):
    """
    Finds all possible pairs of human and model JSON files and categorizing BOTH 
    """
    pairs = defaultdict(lambda: defaultdict(list))
    print(f"Searching for all human/model file pairs...")

    human_category_map = {
        "correct_answers_users": "human_correct",
        "semi_correct_answers_users": "human_semi_correct",
        "wrong_answers_users": "human_wrong"
    }

    for human_folder, human_category_name in human_category_map.items():
        human_cat_dir = os.path.join(human_dir, human_folder)
        if not os.path.isdir(human_cat_dir):
            continue

        for human_file in os.listdir(human_cat_dir):
            if human_file.endswith("_aggregate.json"):
                base_name = human_file.replace("_aggregate.json", "")
                
                model_file_path = None
                model_category_name = None
                for model_folder in ["correct", "semi_correct", "wrong"]:
                    potential_path = os.path.join(model_dir, model_folder, f"{base_name}_model_attention.json")
                    if os.path.exists(potential_path):
                        model_file_path = potential_path
                        model_category_name = model_folder
                        break
                
                if model_file_path:
                    pairs[human_category_name][model_category_name].append({
                        "human_file": os.path.join(human_cat_dir, human_file),
                        "model_file": model_file_path,
                        "image_name": f"{base_name}.png"
                    })
    return pairs


def create_attention_vectors(all_tokens, human_data, model_data):
    """Creates two aligned lists of attention scores for comparison."""
    num_tokens = len(all_tokens)
    human_vector = np.zeros(num_tokens)
    model_vector = np.zeros(num_tokens)
    
    token_to_idx_map = {tuple(token['bounding_box'].values()): i for i, token in enumerate(all_tokens) if token.get('bounding_box')}

    for item in human_data.get("consensus_bounding_boxes", []):
        key = tuple(item.values())[:4]
        if key in token_to_idx_map:
            human_vector[token_to_idx_map[key]] = item.get("consensus_score", 0)

    for item in model_data.get("attention_map", []):
        bbox_list = item.get('bounding_box', [])
        if len(bbox_list) == 4:
            model_key_approx = (bbox_list[0], bbox_list[1], bbox_list[2] - bbox_list[0], bbox_list[3] - bbox_list[1])
            found_key = next((k for k in token_to_idx_map if np.allclose(np.array(k), np.array(model_key_approx), atol=1e-5)), None)
            if found_key:
                model_vector[token_to_idx_map[found_key]] = item.get("attention_score", 0)
            
    return human_vector, model_vector


def categorize_question(question_text):
    """Categorizes question into simple types based on keywords."""
    if not question_text: return "UNKNOWN"
    q_lower = question_text.lower()
    if any(word in q_lower for word in ['date', 'day', 'year', 'total', 'amount', 'number', 'how many', 'what is the number', 'index']): return "NUMERIC"
    elif any(word in q_lower for word in ['who', 'name', 'company', 'agency', 'clinic', 'brand']): return "ENTITY"
    else: return "OTHER"


if __name__ == "__main__":
    print("Starting Full Matrix Comparison Analysis...")
    
    #data loading
    print(f"Loading OCR data from {OCR_DATA_PATH}...")
    try:
        ocr_data = np.load(OCR_DATA_PATH, allow_pickle=True)
    except FileNotFoundError:
        print(f"FATAL ERROR: OCR data file not found at {OCR_DATA_PATH}"); exit()
    print("Pre-processing OCR data for fast lookups...")
    ocr_map = {os.path.splitext(record['image_name'])[0]: record for record in ocr_data if isinstance(record, dict) and 'image_name' in record}
    print(f"OCR map created with {len(ocr_map)} entries.")
    
    #all matching file pairs
    file_pairs = find_file_pairs(AGGREGATED_HUMAN_JSON_DIR, MODEL_JSON_DIR)
    
   
    correlations_by_model_cat = defaultdict(list)
    correlations_by_q_type = defaultdict(list)
    overlap_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    #data analysis loop
    for human_category, model_categories_dict in file_pairs.items():
        for model_category, pairs_in_category in model_categories_dict.items():
            if not pairs_in_category: continue
            
            print(f"\nProcessing {len(pairs_in_category)} pairs for (Human: {human_category}, Model: {model_category})")
            for image_pair in pairs_in_category:
                with open(image_pair['human_file'], 'r') as f: human_json = json.load(f)
                with open(image_pair['model_file'], 'r') as f: model_json = json.load(f)

                base_name = image_pair['image_name'].replace(".png", "")
                record = ocr_map.get(base_name)
                if not record: continue
                
                all_tokens, question_text = record.get('ocr_info'), record.get('question')
                if not all_tokens: continue

                #spearman corr
                h_vec, m_vec = create_attention_vectors(all_tokens, human_json, model_json)
                if np.std(h_vec) > 0 and np.std(m_vec) > 0:
                    corr, _ = spearmanr(h_vec, m_vec)
                    correlations_by_model_cat[model_category].append(corr)
                    q_type = categorize_question(question_text)
                    correlations_by_q_type[q_type].append(corr)
                
                #Top-N att overlap
                token_to_idx_map = {tuple(token['bounding_box'].values()): i for i, token in enumerate(all_tokens) if token.get('bounding_box')}
                for N in [3, 5, 10]:
                    human_top_n_indices = {token_to_idx_map.get(tuple(item.values())[:4]) for item in human_json.get("consensus_bounding_boxes", [])[:N] if tuple(item.values())[:4] in token_to_idx_map}

                    model_top_n_indices = set()
                    for item in model_json.get("attention_map", [])[:N]:
                        bbox_list = item.get('bounding_box', [])
                        if len(bbox_list) == 4:
                            model_key_approx = (bbox_list[0], bbox_list[1], bbox_list[2] - bbox_list[0], bbox_list[3] - bbox_list[1])
                            found_key = next((k for k in token_to_idx_map if np.allclose(np.array(k), np.array(model_key_approx), atol=1e-5)), None)
                            if found_key: model_top_n_indices.add(token_to_idx_map[found_key])
                    
                    intersection = human_top_n_indices.intersection(model_top_n_indices)
                    overlap_percentage = (len(intersection) / float(N)) * 100
                    
                    overlap_scores[N][human_category][model_category].append(overlap_percentage)

    #results in .txt file
    original_stdout = sys.stdout  #ref original standard output
    with open(REPORT_FILENAME, 'w') as f:
        sys.stdout = f  

        print("--- UNIFIED ANALYSIS REPORT ---")
        
        print("\n--- 1. Overall Correlation by Model Answer Category ---")
        for category, scores in sorted(correlations_by_model_cat.items()):
            avg_corr = np.mean(scores) if scores else 0
            print(f"    Average Correlation for model '{category.upper()}' cases: {avg_corr:.4f} (from {len(scores)} samples)")
            
        print("\n--- 2. Attention Correlation by Question Type (Aggregated across all comparisons) ---")
        for q_type, scores in sorted(correlations_by_q_type.items()):
            avg_corr = np.mean(scores) if scores else 0
            print(f"    Average Correlation for '{q_type}' questions: {avg_corr:.4f} (from {len(scores)} samples)")
            
        print("\n--- 3. Graduated Word Attention Overlap ---")
        if not overlap_scores:
            print("    No data available for Top-N Overlap analysis.")
        else:
            for N in sorted(overlap_scores.keys()):
                print(f"\n    --- Top-{N} Overlap Analysis ---")
                for human_category, model_results in sorted(overlap_scores[N].items()):
                    print(f"        Baseline: {human_category.replace('_', ' ').title()}")
                    if not model_results:
                         print("            No model data for this baseline.")
                    for model_category, scores in sorted(model_results.items()):
                        avg_overlap = np.mean(scores) if scores else 0
                        print(f"            vs. Model '{model_category.upper()}' cases: {avg_overlap:.2f}% overlap (from {len(scores)} samples)")

    sys.stdout = original_stdout  #reset std output to og value
    print(f"\nAnalysis complete. Report saved to '{REPORT_FILENAME}'")