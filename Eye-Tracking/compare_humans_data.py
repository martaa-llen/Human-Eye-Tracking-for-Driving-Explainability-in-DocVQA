import os
import json
import numpy as np
import textwrap
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

#CONF
INPUT_DATA_DIR = "2_bounding_boxes_heatmaps/json"
OUTPUT_DIR = "consensus_human_results"
DOC_IMAGE_DIR = r"D:\tfg\TFG_FINAL\dataset_landscape" # "C:/Users/marta/tfg/dataset_landscape/" 
OCR_DATA_PATH = r"D:\tfg\TFG_FINAL\filtered_imdb_train_landscape.npy" #"C:/Users/marta/tfg/filtered_imdb_train_landscape.npy"
CONSENSUS_THRESHOLD = 0.55

print(f"Loading OCR data from {OCR_DATA_PATH}...")
ocr_data = np.load(OCR_DATA_PATH, allow_pickle=True)
print("OCR data loaded.")


def discover_files(input_dir):
    """Finds all JSON files and groups them by image name and the correct category folder names."""
    
    grouped_files = {
        "correct_answers_users": defaultdict(list),
        "semi_correct_answers_users": defaultdict(list),
        "wrong_answers_users": defaultdict(list)
    }
    
    print(f"Searching for JSON files in: {input_dir}")
    if not os.path.isdir(input_dir):
        print(f"--- ERROR: Input directory not found at '{input_dir}'. Please check the path. ---")
        return None

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith("_human_attention.json"):
                category = os.path.basename(root)
                if category in grouped_files:
                    suffix_to_remove = "_human_attention.json"
                    image_name_with_ext = file[:-len(suffix_to_remove)]
                    grouped_files[category][image_name_with_ext].append(os.path.join(root, file))

    print(f"Found data for {len(grouped_files['correct_answers_users'])} images with correct answers.")
    print(f"Found data for {len(grouped_files['semi_correct_answers_users'])} images with semi-correct answers.")
    print(f"Found data for {len(grouped_files['wrong_answers_users'])} images with wrong answers.")
    return grouped_files


def plot_aggregated_heatmap(image_path, consensus_bboxes, output_path, question_text, category, answers_to_display, correct_answer):
    """Draws consensus heatmap with a header for the question and a detailed footer for the answers."""
    try:
        base_image = Image.open(image_path)
        image_width, image_height = base_image.size

        # Setup fonts 
        header_font_size = max(20, int(image_width / 35))
        footer_font_size = max(18, int(image_width / 40))
        try:
            header_font = ImageFont.truetype("arial.ttf", size=header_font_size)
            footer_font = ImageFont.truetype("arial.ttf", size=footer_font_size)
        except IOError:
            header_font = ImageFont.load_default()
            footer_font = ImageFont.load_default()

        #header text
        char_width_header = header_font.getbbox("a")[2] if hasattr(header_font, 'getbbox') else 10
        header_wrap_width = int((image_width - 30) / char_width_header)
        wrapped_question = textwrap.fill(f"Question: {question_text}", width=header_wrap_width)
        
        temp_draw = ImageDraw.Draw(Image.new('RGB', (0,0)))
        header_text_bbox = temp_draw.multiline_textbbox((0, 0), wrapped_question, font=header_font, spacing=5)
        header_height = (header_text_bbox[3] - header_text_bbox[1]) + 40

        #footer text
        char_width_footer = footer_font.getbbox("a")[2] if hasattr(footer_font, 'getbbox') else 10
        footer_wrap_width = int((image_width - 30) / char_width_footer)
        
        footer_lines = []
        if category in ["wrong_answers_users", "semi_correct_answers_users"]:
            answers_str = ", ".join(answers_to_display)
            footer_lines.append(textwrap.fill(f"Submitted Answers: {answers_str}", width=footer_wrap_width))
            footer_lines.append(textwrap.fill(f"Correct Answer: {correct_answer}", width=footer_wrap_width))
        else: #'correct_answers_users'
            footer_lines.append(textwrap.fill(f"Correct Answer: {correct_answer}", width=footer_wrap_width))
        
        footer_text = "\n".join(footer_lines)
        footer_text_bbox = temp_draw.multiline_textbbox((0, 0), footer_text, font=footer_font, spacing=5)
        footer_height = (footer_text_bbox[3] - footer_text_bbox[1]) + 30

        #canvas and draw elements
        final_canvas = Image.new('RGB', (image_width, header_height + image_height + footer_height), 'white')
        canvas_draw = ImageDraw.Draw(final_canvas)
        canvas_draw.multiline_text((15, 20), wrapped_question, font=header_font, fill='black', spacing=5)
        final_canvas.paste(base_image, (0, header_height))
        canvas_draw.multiline_text((15, header_height + image_height + 15), footer_text, font=footer_font, fill='black', spacing=5)

        #draw heatmap overlay
        overlay = Image.new("RGBA", base_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        heatmap_color = (255, 140, 0, 100) 
        for bbox_dict in consensus_bboxes:
            x1 = bbox_dict['topLeftX'] * image_width
            y1 = bbox_dict['topLeftY'] * image_height
            x2 = x1 + (bbox_dict['width'] * image_width)
            y2 = y1 + (bbox_dict['height'] * image_height)
            draw.rectangle([x1, y1, x2, y2], fill=heatmap_color)
        
        final_canvas.paste(overlay, (0, header_height), mask=overlay)
        final_canvas.save(output_path)
        print(f"Saved aggregated plot: {output_path}")

    except FileNotFoundError:
        print(f"--- ERROR: Original document image not found at '{image_path}'. Skipping plot. ---")
    except Exception as e:
        print(f"--- ERROR: Could not create plot for '{image_path}': {e} ---")


def process_image_group(category, image_name, user_json_paths):
    """
    Processes all user data for a single image and now correctly calls the plotting function.
    """
    print(f"\nProcessing: {image_name} (Category: {category}, Users: {len(user_json_paths)})")
    
    num_users = len(user_json_paths)
    bbox_counts = defaultdict(int)
    all_user_answers = []
    
    #init all metadata variables
    correct_answer, image_number, question_text = "N/A", -1, "Question not found"

    #data from all users
    for user_file in user_json_paths:
        with open(user_file, 'r') as f:
            data = json.load(f)

        #metadata from the first file we process
        if image_number == -1: 
            correct_answer = data.get("correct_answer", "N/A")
            image_number = data.get("image_number", -1) 
            if image_number != -1:
                try:
                    question_text = ocr_data[image_number]['question']
                except IndexError:
                    print(f"--- WARNING: image_number {image_number} is out of bounds for ocr_data. ---")
                    question_text = "Question lookup failed"

        if category in ["wrong_answers_users", "semi_correct_answers_users"]:
            all_user_answers.append(data.get("user_answer", "N/A"))

        attention_list = data.get("pre_typing_attention", [])
        unique_bboxes_for_user = set()
        for item in attention_list:
            bbox = item.get('bounding_box', {})
            if bbox:
                bbox_tuple = (bbox['topLeftX'], bbox['topLeftY'], bbox['width'], bbox['height'])
                unique_bboxes_for_user.add(bbox_tuple)
        for bbox_tuple in unique_bboxes_for_user:
            bbox_counts[bbox_tuple] += 1
            
    #consensus bounding boxes
    consensus_bboxes = []
    for bbox_tuple, count in bbox_counts.items():
        if (count / num_users) >= CONSENSUS_THRESHOLD:
            consensus_bboxes.append({'topLeftX': bbox_tuple[0], 'topLeftY': bbox_tuple[1],'width': bbox_tuple[2], 'height': bbox_tuple[3],'consensus_score': count / num_users})
    
    consensus_bboxes.sort(key=lambda x: x['consensus_score'], reverse=True)
    print(f"Found {len(consensus_bboxes)} consensus areas.")
    if not consensus_bboxes:
        print("No consensus areas found for this image, skipping file generation.")
        return

    #save aggregated JSON
    final_json_data = {"image_name": image_name, "category": category, "user_count": num_users, "consensus_threshold": CONSENSUS_THRESHOLD, "consensus_bounding_boxes": consensus_bboxes}
    unique_user_answers = list(set(all_user_answers))
    if category == "correct_answers_users":
        final_json_data["correct_answer"] = correct_answer
    else: 
        final_json_data["submitted_answers"] = unique_user_answers
        final_json_data["correct_answer"] = correct_answer

    json_output_dir = os.path.join(OUTPUT_DIR, "json", category)
    os.makedirs(json_output_dir, exist_ok=True)
    json_path = os.path.join(json_output_dir, f"{os.path.splitext(image_name)[0]}_aggregate.json")
    with open(json_path, 'w') as f:
        json.dump(final_json_data, f, indent=4)
    print(f"Saved aggregated JSON: {json_path}")

    #save the aggregated plot
    plot_output_dir = os.path.join(OUTPUT_DIR, "plots", category)
    os.makedirs(plot_output_dir, exist_ok=True)
    original_image_path = os.path.join(DOC_IMAGE_DIR, f"{image_name}.png")
    plot_path = os.path.join(plot_output_dir, f"{os.path.splitext(image_name)[0]}_aggregate.png")

    plot_aggregated_heatmap(
        image_path=original_image_path,
        consensus_bboxes=consensus_bboxes,
        output_path=plot_path,
        question_text=question_text,
        category=category,
        answers_to_display=unique_user_answers,
        correct_answer=correct_answer
    )

if __name__ == "__main__":
    grouped_files = discover_files(INPUT_DATA_DIR)
    if grouped_files:
        for category, image_map in grouped_files.items():
            if not image_map:
                print(f"--- No files found for category: {category} ---")
                continue
            for image_name, json_paths in image_map.items():
                process_image_group(category, image_name, json_paths)
    print("\n--- Aggregation analysis complete. ---")
    print(f"Results saved in '{OUTPUT_DIR}' folder.")