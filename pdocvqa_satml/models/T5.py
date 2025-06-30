import random
import models._model_utils as model_utils
from models._modules import to_lora_model
from transformers import T5Tokenizer, T5ForConditionalGeneration
import transformers.models.t5.modeling_t5

class T5:
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.tokenizer = T5Tokenizer.from_pretrained(config.model_weights)
        self.model = T5ForConditionalGeneration.from_pretrained(config.model_weights)
        if config.lora:
            self.model = to_lora_model(self.model, config.lora_params)

    def prepare_inputs_for_vqa(self, question, context, answers=None):
        input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip(question, context)]
        tokens = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(self.model.device)

        if answers is not None:
            answers = [random.choice(answer) for answer in answers]
            labels = self.tokenizer(answers, return_tensors='pt', padding=True)
            labels.input_ids[labels.input_ids[:] == self.tokenizer.pad_token_id] = -100
            labels = labels.input_ids.to(self.model.device)
        else:
            labels = None

        return tokens.input_ids, tokens.attention_mask, labels

    def forward(self, batch, return_pred_answer=False):
        question = batch['questions']
        context = batch['contexts']
        answers = batch['answers']

        input_ids, attention_mask, labels = self.prepare_inputs_for_vqa(question, context, answers)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        pred_answers, pred_answers_conf = self.get_answer_from_model_output(input_ids, attention_mask) if return_pred_answer else None

        return outputs, pred_answers, pred_answers_conf

    def get_answer_from_model_output(self, input_ids, attention_mask):
        output = self.model.generate(inputs=input_ids, attention_mask=attention_mask, output_scores=True, return_dict_in_generate=True, output_attentions=True)
        pred_answers = self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)
        pred_answers_conf = model_utils.get_generative_confidence(output)

        return pred_answers, pred_answers_conf
