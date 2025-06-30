import random
import numpy as np

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import PreTrainedModel
from models._modules import CustomT5Config, SpatialEmbeddings, VisualEmbeddings # , to_lora_model
import models._model_utils as model_utils
import transformers.models.t5.modeling_t5

class HF_VT5(PreTrainedModel):
    def __init__(self, t5_config):
        super().__init__(t5_config)
        self.language_backbone = T5ForConditionalGeneration(t5_config)
        self.spatial_embedding = SpatialEmbeddings(t5_config)
        self.visual_embedding = VisualEmbeddings(t5_config)

class VT5:
    def __init__(self, config):
        self.max_input_tokens = getattr(config, 'max_input_tokens', 512)

        t5_config = CustomT5Config.from_pretrained(config.model_weights)
        t5_config.visual_module_config = config.visual_module
        self.tokenizer = T5Tokenizer.from_pretrained(config.model_weights)
        self.model = HF_VT5.from_pretrained(config.model_weights, config=t5_config)
        # if config.lora:
        #     self.model = to_lora_model(self.model, config.lora_params)

    def prepare_inputs_for_vqa(self, question, words, boxes, images, answers=None):
        bs = len(words)
        # input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip(question, words)]
        prompt_text = ["question: {:s}  context: ".format(q) for q in question]
        prompt_box = [0, 0, 1000, 1000]
        eos_box = [0, 0, 0, 0]
        padding_box_value = 0  # To become [0, 0, 0, 0] array.

        # Get input_ids, attention_mask and boxes.
        longest_seq = 0
        batch_input_ids = []
        batch_input_boxes = []
        
        batch_prompt_lengths = []

        for batch_idx in range(bs):
            tokenized_prompt = self.tokenizer(prompt_text[batch_idx])
            input_ids = tokenized_prompt.input_ids[:-1]
            input_boxes = [prompt_box] * len(input_ids)

            prompt_len = len(input_ids)
            batch_prompt_lengths.append(prompt_len)

            for word, box in zip(words[batch_idx], boxes[batch_idx]):
                tokenized_word = self.tokenizer(word).input_ids[:-1]  # Tokenize the word and ignore eos_token
                input_ids.extend(tokenized_word)
                input_boxes.extend([box]*len(tokenized_word))  # Repeat the box for each token corresponding to the word.

            batch_input_ids.append(input_ids[:self.max_input_tokens-1] + [self.tokenizer.eos_token_id])  # Append the eos_token at the end.
            #batch_input_boxes.append(np.concatenate([input_boxes[:self.max_input_tokens-1],  np.array([eos_box])]))  # Append a bounding box corresponding to the eos_token.
            batch_input_boxes.append(np.array(input_boxes[:self.max_input_tokens-1] + [eos_box]))
            longest_seq = min(max(longest_seq, len(input_ids) + 1), self.max_input_tokens)

        # Convert to tensors and pad. Actually, a pad tensor is created and it's filled with corresponding values.
        tensor_input_ids = torch.full([bs, longest_seq], fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        tensor_boxes = torch.full([bs, longest_seq, 4],  fill_value=padding_box_value, dtype=torch.float)
        tensor_attention_mask = torch.zeros([bs, longest_seq], dtype=torch.long)

        tensor_prompt_lengths = torch.tensor(batch_prompt_lengths, dtype=torch.long)

        for batch_idx in range(bs):
            tensor_input_ids[batch_idx, :len(batch_input_ids[batch_idx])] = torch.LongTensor(batch_input_ids[batch_idx])
            tensor_boxes[batch_idx, :len(batch_input_boxes[batch_idx])] = torch.from_numpy(batch_input_boxes[batch_idx][:len(batch_input_boxes[batch_idx])])
            tensor_attention_mask[batch_idx, :len(batch_input_ids[batch_idx])] = 1

        # Send everything to GPU
        tensor_input_ids = tensor_input_ids.to(self.model.device)
        tensor_boxes = tensor_boxes.to(self.model.device)
        tensor_attention_mask = tensor_attention_mask.to(self.model.device)

        # Get semantic and spatial embeddings
        semantic_embedding = self.model.language_backbone.shared(tensor_input_ids)
        spatial_embedding = self.model.spatial_embedding(tensor_boxes)
        
        visual_embedding, visual_emb_mask = self.model.visual_embedding(images)
        visual_boxes = self.model.visual_embedding.get_visual_boxes(num_pages=bs).to(self.model.device)

        # input_embeds = semantic_embedding
        input_embeds = torch.add(semantic_embedding, spatial_embedding)
        input_embeds = torch.cat([input_embeds, visual_embedding], dim=1)  # Concatenate semantic + visual embeddings TODO: Provide visual bounding boxes.
        tensor_attention_mask = torch.cat([tensor_attention_mask, visual_emb_mask], dim=1)

        # Tokenize answers
        if answers is not None:
            answers = [random.choice(answer) for answer in answers]
            labels = self.tokenizer(answers, return_tensors='pt', padding=True)
            labels.input_ids[labels.input_ids[:] == self.tokenizer.pad_token_id] = -100
            labels = labels.input_ids.to(self.model.device)
        else:
            labels = None
        mapping_info = (tensor_input_ids, tensor_boxes, longest_seq, tensor_prompt_lengths, visual_boxes)
        return input_embeds, tensor_attention_mask, labels, mapping_info

    def forward(self, batch, return_pred_answer=False):
        question = batch['questions']
        words = batch['words']
        boxes = batch['boxes']
        images = batch['images']
        answers = batch['answers']

        #input_embeds, attention_mask, labels = self.prepare_inputs_for_vqa(question, words, boxes, images, answers)
        input_embeds, attention_mask, labels, mapping_info = self.prepare_inputs_for_vqa(question, words, boxes, images, answers)

        outputs = self.model.language_backbone(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels, output_attentions=True)
        if return_pred_answer:
            pred_answers, prediction_outputs = self.get_answer_from_model_output(input_embeds, attention_mask)
        else:
            pred_answers, prediction_outputs =  None, None

        return outputs, pred_answers, prediction_outputs, mapping_info

    def get_answer_from_model_output(self, input_embeds, attention_mask):
        output = self.model.language_backbone.generate(inputs_embeds=input_embeds, attention_mask=attention_mask, output_scores=True, return_dict_in_generate=True, output_attentions=True)
        pred_answers = self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)

        output_masks = (output.sequences[:, 1:] == self.tokenizer.pad_token_id) | (output.sequences[:, 1:] == self.tokenizer.eos_token_id) # b x s 
        logits, confidences = model_utils.get_generative_confidence(output, output_masks)
   
        return pred_answers, (logits, confidences, output.decoder_attentions, output.get('cross_attentions'))
