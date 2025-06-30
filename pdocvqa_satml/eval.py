import os, time
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.PFL_DocVQA import collate_fn

from logger import Logger
from metrics import Evaluator
from utils import parse_args, time_stamp_to_hhmmss, load_config, save_json
from build_utils import build_model, build_dataset
from checkpoint import save_model

import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

from difflib import SequenceMatcher

import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm
import numpy as np
import textwrap
import json
import traceback


def save_attention_visualization(
        pil_image,
        filename,
        question_text="",
        predicted_answer="",
        correct_answer=None,
        similarity_answers=None,
        text_boxes=None,
        text_attention_scores=None,
        visual_boxes=None,
        visual_attention_scores=None):
    """
    Saves a visualization of attention scores
    """
    try:
        if not isinstance(pil_image, Image.Image):
            print(f"Error: Image is not a PIL.Image object for {filename}. Skipping.")
            return

        #setup canvas, header and footer 
        image_width, image_height = pil_image.size
        FONT_SIZE_RATIO = 25
        #font size
        header_font_size = max(20, min(45, int(image_width / FONT_SIZE_RATIO)))
        footer_font_size = max(18, int(header_font_size * 0.85))
       
        try:
            header_font = ImageFont.truetype("arial.ttf", size=header_font_size)
            footer_font = ImageFont.truetype("arial.ttf", size=footer_font_size)
        except IOError:
            header_font = ImageFont.load_default()
            footer_font = ImageFont.load_default()
        
        #add question in header
        header_wrap_width = max(20, int(image_width / (header_font.getbbox("a")[2] if hasattr(header_font, 'getbbox') else 10) * 1.5))
        header_text = textwrap.fill(f"Question: {question_text}", width=header_wrap_width)
        
        temp_draw = ImageDraw.Draw(Image.new('RGB', (0,0)))
        header_text_bbox = temp_draw.multiline_textbbox((0, 0), header_text, font=header_font, spacing=5)
        header_height = (header_text_bbox[3] - header_text_bbox[1]) + 40
        
        correct_answer_str = str(correct_answer) if correct_answer is not None else "N/A"
        similarity_str = f"{similarity_answers:.3f}" if similarity_answers is not None else "N/A"
        
        #add pred answ, correct answ and similarity in footer
        footer_wrap_width = max(20, int(image_width / (footer_font.getbbox("a")[2] if hasattr(footer_font, 'getbbox') else 10) * 1.5))
        footer_text = (textwrap.fill(f"Predicted Answer: {predicted_answer}", width=footer_wrap_width) + "\n" +
                       textwrap.fill(f"Correct Answer: {correct_answer_str}", width=footer_wrap_width) + "\n" +
                       textwrap.fill(f"Similarity: {similarity_str}", width=footer_wrap_width))
        
        footer_text_bbox = temp_draw.multiline_textbbox((0, 0), footer_text, font=footer_font, spacing=5)
        footer_height = (footer_text_bbox[3] - footer_text_bbox[1]) + 40
        
        final_canvas = Image.new('RGB', (image_width, image_height + header_height + footer_height), 'white')
        canvas_draw = ImageDraw.Draw(final_canvas)
        canvas_draw.multiline_text((15, 20), header_text, font=header_font, fill='black', spacing=5)
        
        footer_y_start = header_height + image_height + 20
        canvas_draw.line([(0, header_height + image_height + 1), (image_width, header_height + image_height + 1)], fill=(200, 200, 200), width=2)
        canvas_draw.multiline_text((15, footer_y_start), footer_text, font=footer_font, fill='black', spacing=5)


        #overlays 
        base_image = pil_image.convert('RGBA')
        text_heatmap_overlay = Image.new('RGBA', pil_image.size, (255, 255, 255, 0))
        visual_heatmap_overlay = Image.new('RGBA', pil_image.size, (255, 255, 255, 0))

        #draw TEXT attention
        if text_attention_scores is not None and text_boxes is not None and len(text_attention_scores) > 0:
            text_heatmap_draw = ImageDraw.Draw(text_heatmap_overlay)
            
            #relative normalization 
            max_score = text_attention_scores.max()
            if max_score > 0:
                normalized_scores = text_attention_scores / max_score
            else:
                normalized_scores = text_attention_scores

            text_colormap = cm.get_cmap('Reds')
            for i in range(len(text_boxes)):
                box = text_boxes[i].tolist()
                abs_box = [int(box[0] * image_width), int(box[1] * image_height), int(box[2] * image_width), int(box[3] * image_height)]
                color_float = text_colormap(normalized_scores[i])
                fill_color = (int(color_float[0]*255), int(color_float[1]*255), int(color_float[2]*255), 150)
                text_heatmap_draw.rectangle(abs_box, fill=fill_color)

        #draw VISUAL attention 
        if visual_attention_scores is not None and visual_boxes is not None and len(visual_attention_scores) > 0:
            visual_heatmap_draw = ImageDraw.Draw(visual_heatmap_overlay)
            max_score, min_score = visual_attention_scores.max(), visual_attention_scores.min()
            if max_score > min_score:
                norm_scores = (visual_attention_scores - min_score) / (max_score - min_score)
            else:
                norm_scores = np.zeros_like(visual_attention_scores)
            
            visual_colormap = cm.get_cmap('Blues')
            for i in range(len(visual_boxes)):
                box_as_list = visual_boxes[i].tolist()
                abs_box = [
                    int(box_as_list[0] * image_width), int(box_as_list[1] * image_height),
                    int(box_as_list[2] * image_width), int(box_as_list[3] * image_height)
                ]
                color_float = visual_colormap(norm_scores[i])
                fill_color = (int(color_float[0] * 255), int(color_float[1] * 255), int(color_float[2] * 255), 140)
                visual_heatmap_draw.rectangle(abs_box, fill=fill_color)

        #composite Layers
        composited_image = Image.alpha_composite(base_image, visual_heatmap_overlay)
        composited_image = Image.alpha_composite(composited_image, text_heatmap_overlay)
        
        final_canvas.paste(composited_image.convert('RGB'), (0, header_height))
        final_canvas.save(filename)

    except Exception as e:
        print(f"--- FATAL ERROR during visualization for {filename} ---")
        traceback.print_exc()

def save_attention_rollout_image(attn_scores, filename):
    """
    Saves bar chart visualization of 1D attention scores.
    """
    if attn_scores.ndim != 1:
        print(f"Warning: Expected 1D array for rollout plot, but got shape {attn_scores.shape}. Skipping plot.")
        return
        
    plt.figure(figsize=(12, 6)) 
    token_indices = np.arange(len(attn_scores))
    plt.bar(token_indices, attn_scores)
    plt.xlabel("Input Token Index")
    plt.ylabel("Attention Score")
    plt.title("Attention Rollout Scores per Input Token")
    plt.xlim(0, len(attn_scores)) 
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def anls(pred_answer, real_answers):
    """
    Function to compare user answer with the real answer and return the similarity between them
    Params:
        - image_number (int): used to locate image information in dataset
        - user_answer (str): the answer given by the user

    Returns:
        - max_similarity (float): the maximum similarity between the user answer and the real answer
        - answer_matched (str): the real answer that matched the user answer
    """
    #function to compare user answer with the real answer 
    max_similarity = 0.0

    answer_matched = None

    if not real_answers: 
        return 0.0, 0.0
    
    for correct_answer in real_answers: 
        similarity = SequenceMatcher(None, pred_answer.lower(), correct_answer.lower()).ratio()
        if similarity > max_similarity: 
            max_similarity = similarity
            answer_matched = correct_answer


    return max_similarity, answer_matched

def do_boxes_overlap(box_a, box_b):
    """
    Checks if two bounding boxes overlap.
    Box format is assumed to be [x_min, y_min, x_max, y_max].
    """
    #check if rectangle is to the left of the other
    if box_a[2] < box_b[0] or box_b[2] < box_a[0]:
        return False
    
    #check if rectangle is above other
    if box_a[3] < box_b[1] or box_b[3] < box_a[1]:
        return False
        
    return True

def evaluate(data_loader, model, evaluator, config, epoch, max_steps=None,
              save_plots=True, should_save_json=False, use_human_occlusion=False, 
              save_plots_occluded=False, use_opposite_occlusion=False, evaluate_first_question_only=True):
    
    scores_by_samples, total_accuracies, total_anls, all_pred_answers_list = {}, [], [], []
    processed_docs = set()
    model.model.eval()
    
    #directories to save data
    main_dir = "Results_prova4"
    os.makedirs(main_dir, exist_ok=True)

    if save_plots:
        base_plot_dir = f"{main_dir}/attention_visualizations"
        os.makedirs(base_plot_dir, exist_ok=True)
    if should_save_json:
        model_attention_dir = f"{main_dir}/model_attention_maps"
        os.makedirs(model_attention_dir, exist_ok=True)
        
    if use_human_occlusion:
        occlusion_output_dir = f"{main_dir}/images_used_occluded"
        os.makedirs(occlusion_output_dir, exist_ok=True)
        not_occluded_dir = f"{occlusion_output_dir}/not_occluded"
        os.makedirs(not_occluded_dir, exist_ok=True)
    
    if use_opposite_occlusion:
        occlusion_output_dir = f"{main_dir}/images_used_opp_occluded"
        os.makedirs(occlusion_output_dir, exist_ok=True)
        not_occluded_dir = f"{occlusion_output_dir}/not_occluded"
        os.makedirs(not_occluded_dir, exist_ok=True)

    for batch_loop_idx, batch in enumerate(tqdm(data_loader, desc=f"Evaluating Epoch {epoch}")):
        
        filtered_batch = {key: value for key, value in batch.items()}
        if evaluate_first_question_only: 
            #human trials just used the first question of each document
            
            #batch filtering
            #identify which samples in the current batch are new.
            indices_to_keep = []
            for i in range(len(batch['question_id'])):
                doc_id = batch['image_names'][i]
                if doc_id not in processed_docs:
                    #first time this document has been seen
                    indices_to_keep.append(i)
                    #add it to the set to handle duplicates within the same batch
                    processed_docs.add(doc_id)

            #skip to the next batch if document already seen
            if not indices_to_keep:
                continue

            #new batch containing only the first question to each document
            filtered_batch = {key: [value[i] for i in indices_to_keep] for key, value in batch.items()}
        
        
        #occlusion logic using filtered_batch (docs and questions that humans have seen)
        if use_human_occlusion or use_opposite_occlusion:
            mode = "opposite" if use_opposite_occlusion else "standard"
            print(f"Running with human data occlusion (Mode: {mode})...")
            
            occluded_images = []
            #store the new, filtered text/box data
            new_words_list = []
            new_boxes_list = []
            for i in range(len(filtered_batch['images'])):
                original_pil_image = filtered_batch['images'][i]
                doc_id = filtered_batch['image_names'][i]
                doc_filename_base = os.path.splitext(os.path.basename(doc_id))[0]
                
                human_json_path = find_human_consensus_file(doc_filename_base)
                human_consensus_bboxes = []
                if human_json_path:
                    with open(human_json_path, 'r') as f:
                        human_data = json.load(f)
                    human_consensus_bboxes = human_data.get("consensus_bounding_boxes", [])
                               
                #gt answer from the filtered batch
                correct_answer_string = filtered_batch['answers'][i][0] if filtered_batch['answers'][i] else None
                
                #try find the answer's location
                answer_bboxes = find_answer_bboxes(correct_answer_string, filtered_batch['words'][i], filtered_batch['boxes'][i], human_consensus_bboxes)
                
               
                #just keep those inside visible region.
                visible_words = []
                visible_boxes = []
                human_consensus_regions = []

                for bbox_dict in human_consensus_bboxes:
                    x1 = bbox_dict['topLeftX']
                    y1 = bbox_dict['topLeftY']
                    x2 = x1 + bbox_dict['width']
                    y2 = y1 + bbox_dict['height']
                    human_consensus_regions.append([x1, y1, x2, y2])

               
                
                if use_opposite_occlusion:
                    #OPPOSITE MODE: see answ + not consensus human
                    if not isinstance(human_consensus_regions, np.ndarray):
                        human_consensus_regions = np.array(human_consensus_regions)

                    for word_idx, word_box in enumerate(filtered_batch['boxes'][i]):
                        #visible word if in answer
                        word_text = filtered_batch['words'][i][word_idx]
                        
                        is_in_answer = False
                        for answer_box in answer_bboxes:
                            if do_boxes_overlap(word_box, answer_box):
                                is_in_answer = True
                                break  #found word in answ

                        #check if word in consensus box
                        is_in_consensus = False
                        for consensus_box in human_consensus_regions:
                            if do_boxes_overlap(word_box, consensus_box):
                                is_in_consensus = True
                                break  

                        should_be_kept = is_in_answer or not is_in_consensus


                        if is_in_answer or not is_in_consensus:
                            visible_words.append(filtered_batch['words'][i][word_idx])
                            visible_boxes.append(word_box)
                else:
                    #vars numpy arrays
                    if not isinstance(human_consensus_regions, np.ndarray):
                        human_consensus_regions = np.array(human_consensus_regions)
                    if not isinstance(answer_bboxes, np.ndarray):
                        answer_bboxes = np.array(answer_bboxes)

                    # prevent mismatch (2D and 1D)
                    if answer_bboxes.size > 0 and answer_bboxes.ndim == 1:
                        answer_bboxes = answer_bboxes.reshape(1, -1)

                    #concatenate two arrays
                    if human_consensus_regions.size > 0 and answer_bboxes.size > 0:
                        visible_regions = np.concatenate((human_consensus_regions, answer_bboxes), axis=0)
                    elif human_consensus_regions.size > 0:
                        visible_regions = human_consensus_regions
                    elif answer_bboxes.size > 0:
                        visible_regions = answer_bboxes
                    else:
                        visible_regions = np.array([]) 

                    if len(visible_regions) > 0:
                        for word_idx, word_box in enumerate(filtered_batch['boxes'][i]):
                            for visible_area in visible_regions:
                                #word box is contained in any visible area.
                                if do_boxes_overlap(word_box, visible_area):
                                    visible_words.append(filtered_batch['words'][i][word_idx])
                                    visible_boxes.append(word_box)
                                    break #found visible area --> go to next word 
                
                new_words_list.append(visible_words)
                new_boxes_list.append(visible_boxes)

                
                #answer not found --> use og image
                if len(answer_bboxes) == 0:
                    print(f"\n--> Could not find location for answer '{correct_answer_string}'. Using ORIGINAL image for {doc_filename_base}.")
                    occluded_images.append(original_pil_image) #original image
                    if save_plots_occluded: 
                        not_occlusion_image_path = os.path.join(not_occluded_dir, f"{doc_filename_base}_occluded.png")
                        original_pil_image.save(not_occlusion_image_path)
                else:
                    #answ found --> create occl image.
                    if use_opposite_occlusion: 
                        occluded_image = create_opposite_occluded_image(original_pil_image, human_consensus_bboxes, answer_bboxes)
                        
                    else: 
                        occluded_image = create_occluded_image(original_pil_image, human_consensus_bboxes, answer_bboxes)
                    
                    occluded_images.append(occluded_image)
                    
                    if save_plots_occluded: 
                        occlusion_image_path = os.path.join(occlusion_output_dir, f"{doc_filename_base}_occluded.png")
                        occluded_image.save(occlusion_image_path)
                
            #replace images in batch with new list images
            filtered_batch['images'] = occluded_images
            filtered_batch['words'] = new_words_list
            filtered_batch['boxes'] = new_boxes_list
        
        with torch.no_grad():
            _, current_batch_predicted_answers, prediction_extras, mapping_info = model.forward(filtered_batch, return_pred_answer=True)
        
        if mapping_info:
             _, current_batch_text_token_boxes, padded_text_len, current_batch_prompt_lengths, current_batch_visual_boxes = mapping_info
        else: 
            _, current_batch_text_token_boxes, padded_text_len, current_batch_prompt_lengths, current_batch_visual_boxes = [None] * 5
        batch_cross_attentions = prediction_extras[-1] if prediction_extras and isinstance(prediction_extras, tuple) else None

        #processes only unique items from filtered batch 
        for i in range(len(filtered_batch['question_id'])):
            doc_id = filtered_batch['image_names'][i]
            question_id = filtered_batch['question_id'][i]
            
            word_level_scores = []
            words = filtered_batch['words'][i]
            word_boxes = filtered_batch['boxes'][i]
            
            if batch_cross_attentions and current_batch_text_token_boxes is not None and current_batch_prompt_lengths is not None:
                last_layer_attentions = batch_cross_attentions[-1][0] if isinstance(batch_cross_attentions[-1], tuple) else batch_cross_attentions[-1]
                token_scores = last_layer_attentions[i].squeeze(1).max(dim=0).values.detach().cpu().numpy()
                token_boxes = current_batch_text_token_boxes[i].cpu().numpy()
                prompt_len = current_batch_prompt_lengths[i].item()
                context_token_scores = token_scores[prompt_len:]
                context_token_boxes = token_boxes[prompt_len:]
                token_centers = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in context_token_boxes]
                
                if token_centers:
                    for word_box in word_boxes:
                        contained_scores = []
                        for token_idx, token_center in enumerate(token_centers):
                            min_x, min_y, max_x, max_y = word_box[0], word_box[1], word_box[2], word_box[3]
                            if (min_x <= token_center[0] <= max_x) and (min_y <= token_center[1] <= max_y):
                                if token_idx < len(context_token_scores):
                                    contained_scores.append(context_token_scores[token_idx])
                        word_level_scores.append(np.max(contained_scores) if contained_scores else 0)

            word_level_scores = np.array(word_level_scores)
            pred_answer = current_batch_predicted_answers[i] if current_batch_predicted_answers else "N/A"
            #gt_answers = batch['answers'][i]
            gt_answers = filtered_batch['answers'][i]
            
            #gt answ in list for anls
            if isinstance(gt_answers, str):
                gt_answers_for_anls = [gt_answers]
            else:
                gt_answers_for_anls = gt_answers
            similarity, answer_matched = anls(pred_answer, gt_answers_for_anls)

            #set category to later save data
            if similarity > 0.9:
                category = "correct"
            elif similarity > 0.5:
                category = "semi_correct"
            else:
                category = "wrong"

            doc_filename_base = os.path.splitext(os.path.basename(doc_id))[0]
            if not evaluate_first_question_only:
                safe_question_id = str(question_id).replace('/', '_').replace('\\', '_')
                doc_filename_base = f"{doc_filename_base}_qid_{safe_question_id}"
      

            if save_plots:
                save_plot_subdir = os.path.join(base_plot_dir, category)
                os.makedirs(save_plot_subdir, exist_ok=True)
                
                visual_scores, visual_boxes = np.array([]), np.array([])
                if batch_cross_attentions and current_batch_visual_boxes is not None and 'token_scores' in locals():
                     raw_visual_scores = token_scores[padded_text_len:]
                     if len(raw_visual_scores) > 0:
                         visual_scores = raw_visual_scores[1:]
                         visual_boxes = current_batch_visual_boxes[i].cpu().numpy()[1:]

            
                common_args = {"pil_image": filtered_batch['images'][i], "question_text": filtered_batch['questions'][i], "predicted_answer": pred_answer, "correct_answer": answer_matched or gt_answers_for_anls[0], "similarity_answers": similarity}
                #common_args = {"pil_image": batch['images'][i], "question_text": batch['questions'][i], "predicted_answer": pred_answer, "correct_answer": answer_matched or gt_answers[0], "similarity_answers": similarity}
                
                #txt-only plot
                save_attention_visualization(filename=os.path.join(save_plot_subdir, f"{doc_filename_base}_text_attn.png"), text_boxes=np.array(word_boxes), text_attention_scores=word_level_scores, **common_args)
                
                #visual-only plot
                save_attention_visualization(filename=os.path.join(save_plot_subdir, f"{doc_filename_base}_visual_attn.png"), visual_boxes=visual_boxes, visual_attention_scores=visual_scores, **common_args)
                
                #combined plot
                save_attention_visualization(filename=os.path.join(save_plot_subdir, f"{doc_filename_base}_combined_attn.png"), text_boxes=np.array(word_boxes), text_attention_scores=word_level_scores, visual_boxes=visual_boxes, visual_attention_scores=visual_scores, **common_args)

                #rollout graph
                if len(word_level_scores) > 0:
                    save_attention_rollout_image(word_level_scores, os.path.join(save_plot_subdir, f"{doc_filename_base}_rollout_graph.png"))
                
                print(f"Saved all plots for: {doc_filename_base}")

            if should_save_json and len(word_level_scores) > 0:
                output_dir = os.path.join(model_attention_dir, category)
                os.makedirs(output_dir, exist_ok=True)
                json_path = os.path.join(output_dir, f"{doc_filename_base}_model_attention.json")
                
                attention_map_list = []
                max_score = word_level_scores.max()
                normalized_scores = word_level_scores / max_score if max_score > 0 else word_level_scores
                
                for idx, word in enumerate(words):
                    if idx < len(normalized_scores):
                        attention_map_list.append({
                            "word": word,
                            "bounding_box": word_boxes[idx].tolist(),
                            "attention_score": normalized_scores[idx].item()
                        })
                
                attention_map_list.sort(key=lambda x: x['attention_score'], reverse=True)
                
                

                output_data_for_json = {
                    "predicted_answer": pred_answer,
                    "correct_answer": answer_matched or gt_answers_for_anls[0],
                    "attention_map": attention_map_list
                }
                
                # print(f"    -> PREPARING JSON with predicted_answer='{output_data_for_json['predicted_answer']}'") # DEBUG
                with open(json_path, 'w') as f:
                    json.dump(output_data_for_json, f, indent=4)
                # print(f"Saved JSON attention map: {json_path}")

            processed_docs.add(doc_id)

        metric = evaluator.get_metrics(filtered_batch['answers'], current_batch_predicted_answers)
        total_accuracies.extend(metric['accuracy'])
        total_anls.extend(metric['anls'])
        if max_steps is not None and batch_loop_idx >= max_steps: break
            
    final_accuracy = np.mean(total_accuracies) if total_accuracies else 0
    final_anls = np.mean(total_anls) if total_anls else 0
            
    return final_accuracy, final_anls, all_pred_answers_list, scores_by_samples

# def do_boxes_overlap(box_a, box_b):
#     """
#     Checks if two bounding boxes in [x_min, y_min, x_max, y_max] format overlap.
#     """
#     # Check if one rectangle is entirely to the left of the other
#     if box_a[2] < box_b[0] or box_b[2] < box_a[0]:
#         return False
    
#     # Check if one rectangle is entirely above the other
#     if box_a[3] < box_b[1] or box_b[3] < box_a[1]:
#         return False
        
#     return True


def find_human_consensus_file(base_name):
    """Searches for a human consensus file in the correct/semi_correct/wrong directories."""
    AGGREGATED_HUMAN_JSON_DIR = r"D:\tfg\TFG_FINAL\Eye-Tracking\consensus_human_results\json" # r"C:\Users\marta\tfg\Eye-Tracking\general_human_results4\json" #"/export/fhome/mllopart/general_human_results4/json"
    for category in ["correct_answers_users", "semi_correct_answers_users", "wrong_answers_users"]:
        path = os.path.join(AGGREGATED_HUMAN_JSON_DIR, category, f"{base_name}_aggregate.json")
        if os.path.exists(path):
            return path
    return None

def find_answer_bboxes(answer_string, all_words, all_boxes, human_consensus_bboxes):
    """
    answer bounding boxes using fuzzy matching. Returns an empty list if no confident match is found.
    """
    if not answer_string:
        return []

    answer_words = [word.strip(".,:;()").lower() for word in answer_string.split()]
    if not answer_words:
        return []

    doc_words_cleaned = [word.strip(".,:;()").lower() for word in all_words]

    best_match_ratio = 0.0
    best_match_start_idx = -1
    
    for i in range(len(doc_words_cleaned) - len(answer_words) + 1):
        doc_subsequence = doc_words_cleaned[i:i + len(answer_words)]
        matcher = SequenceMatcher(None, " ".join(answer_words), " ".join(doc_subsequence))
        ratio = matcher.ratio()
        
        if ratio > best_match_ratio:
            best_match_ratio = ratio
            best_match_start_idx = i

    MATCH_CONFIDENCE_THRESHOLD = 0.85

    if best_match_ratio >= MATCH_CONFIDENCE_THRESHOLD:
        #return corresponding boxes
        return all_boxes[best_match_start_idx : best_match_start_idx + len(answer_words)]
    else:
        #no match good enough --> return empty list
        return []


def create_occluded_image(original_image, consensus_bboxes, answer_bboxes):
    """
    new image blacked out, except for the human consensus regions
    AND the bounding boxes for correct answer.
    """
    occluded_image = Image.new("RGB", original_image.size, (0, 0, 0))
    image_width, image_height = original_image.size
    all_boxes_to_draw = set()

    for bbox_dict in consensus_bboxes:
        key = (bbox_dict['topLeftX'], bbox_dict['topLeftY'], bbox_dict['width'], bbox_dict['height'])
        all_boxes_to_draw.add(key)

    for bbox_array in answer_bboxes:
        key = (bbox_array[0], bbox_array[1], bbox_array[2] - bbox_array[0], bbox_array[3] - bbox_array[1])
        all_boxes_to_draw.add(key)

    for bbox_tuple in all_boxes_to_draw:
        x1 = int(bbox_tuple[0] * image_width)
        y1 = int(bbox_tuple[1] * image_height)
        x2 = int(x1 + (bbox_tuple[2] * image_width))
        y2 = int(y1 + (bbox_tuple[3] * image_height))
        region = original_image.crop((x1, y1, x2, y2))
        occluded_image.paste(region, (x1, y1))
        
    return occluded_image

def create_opposite_occluded_image(original_image, consensus_bboxes, answer_bboxes):
    """
    new image where the human consensus regions are blacked out,
    but the answer area is preserved.
    """
    #copy original img
    occluded_image = original_image.copy()
    draw = ImageDraw.Draw(occluded_image)

    image_width, image_height = original_image.size

    #set tuples answer boxes to check for overlaps
    answer_box_set = set()
    for bbox_array in answer_bboxes:
        
        key = (bbox_array[0], bbox_array[1], bbox_array[2], bbox_array[3])
        answer_box_set.add(key)

    #black out consensus regions (only if not part answer)
    for bbox_dict in consensus_bboxes:
        con_x1 = bbox_dict['topLeftX']
        con_y1 = bbox_dict['topLeftY']
        con_x2 = con_x1 + bbox_dict['width']
        con_y2 = con_y1 + bbox_dict['height']
        
        consensus_key = (con_x1, con_y1, con_x2, con_y2)

        is_answer_box = False
        for answer_key in answer_box_set:
             
            if np.allclose(np.array(consensus_key), np.array(answer_key), atol=1e-5):
                is_answer_box = True
                break
        
        if not is_answer_box:
            #not part answ --> black rectangle over it
            pixel_box = [int(con_x1 * image_width), int(con_y1 * image_height),
                         int(con_x2 * image_width), int(con_y2 * image_height)]
            draw.rectangle(pixel_box, fill=(0, 0, 0))
        
    return occluded_image

def main_eval(config, save_plots=True, should_save_json=False, use_human_occlusion=False, save_plots_occluded=False, 
              use_opposite_occlusion=False, evaluate_first_question_only=False):
    start_time = time.time()

    config.return_scores_by_sample = True
    config.return_answers = True

    dataset = build_dataset(config, 'filtered_imdb_train_landscape') ## TODO: CHECK THIS!
    sampler = None
    pin_memory = False

    val_data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=pin_memory, sampler=sampler)

    model = build_model(config)

    logger = Logger(config=config)
    logger.log_model_parameters(model)

    evaluator = Evaluator(case_sensitive=False)
    #accuracy_list, anls_list, pred_answers, scores_by_samples = evaluate(val_data_loader, model, evaluator, config, epoch=0)
    accuracy_list, anls_list, pred_answers, scores_by_samples = evaluate(val_data_loader, model, evaluator, config, epoch=0, 
                                                                         save_plots=save_plots, should_save_json=should_save_json, 
                                                                         use_human_occlusion=use_human_occlusion, save_plots_occluded=save_plots_occluded, 
                                                                         use_opposite_occlusion=use_opposite_occlusion, 
                                                                         evaluate_first_question_only=evaluate_first_question_only)
    accuracy, anls = np.mean(accuracy_list), np.mean(anls_list)

    inf_time = time_stamp_to_hhmmss(time.time() - start_time, string=True)
    logger.log_val_metrics(accuracy, anls, update_best=False)

    save_data = {
        "Model": config.model_name,
        "Model_weights": config.model_weights,
        "Dataset": config.dataset_name,
        "Page retrieval": getattr(config, 'page_retrieval', '-').capitalize(),
        "Inference time": inf_time,
        "Mean accuracy": accuracy,
        "Mean ANLS": anls,
        "Scores by samples": scores_by_samples,
    }

    results_file = os.path.join(config.save_dir, 'results', config.experiment_name)
    save_json(results_file, save_data)

    print("Results correctly saved in: {:s}".format(results_file))


if __name__ == '__main__':

    #as human trials did just use the first question of each document 
    #we have to set this to true if we want to compare them  with those exact docs-question pairs
    EVALUATE_FIRST_QUESTION_ONLY = True

    #set only one occlusion parameter to true
    USE_HUMAN_OCCLUSION = False
    USE_HUMAN_OCCLUSION_OPPOSITE = True

    if USE_HUMAN_OCCLUSION or USE_HUMAN_OCCLUSION_OPPOSITE:
        #set to True as my human trials just used the first question of each document
        EVALUATE_FIRST_QUESTION_ONLY = True
    
    SAVE_PLOTS_OCCLUDED = True #set to True if need to see occlusion images

    SAVE_PLOTS = True
    SHOULD_SAVE_JSON = True

    args = parse_args()
    config = load_config(args)

    main_eval(config, save_plots=SAVE_PLOTS, should_save_json=SHOULD_SAVE_JSON, 
              use_human_occlusion=USE_HUMAN_OCCLUSION, save_plots_occluded=SAVE_PLOTS_OCCLUDED, 
              use_opposite_occlusion=USE_HUMAN_OCCLUSION_OPPOSITE, 
              evaluate_first_question_only=EVALUATE_FIRST_QUESTION_ONLY)
