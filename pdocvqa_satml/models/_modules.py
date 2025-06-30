import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5Config
from transformers import AutoFeatureExtractor, AutoModel
# from peft import LoraConfig, TaskType
# from peft import get_peft_model

class CustomT5Config(T5Config):
    def __init__(self, max_2d_position_embeddings=1024,  **kwargs):
        super().__init__(**kwargs)
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.hidden_dropout_prob = 0.1
        self.layer_norm_eps = 1e-12

class SpatialEmbeddings(nn.Module):
    """
    Spatial embedding by summing x, y, w, h projected by nn.Embedding to hidden size.
    """

    def __init__(self, config):
        super(SpatialEmbeddings, self).__init__()

        self.x_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.y_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        # self.h_position_embeddings = nn.Embedding(
        #     config.max_2d_position_embeddings, config.hidden_size
        # )
        # self.w_position_embeddings = nn.Embedding(
        #     config.max_2d_position_embeddings, config.hidden_size
        # )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.spatial_emb_matcher = MLP(config.hidden_size, 0, config.hidden_size, 1)

        self.config = config

    def forward(self, bbox):
        """
        Converts floating point bounding box coordinates to integer indices
        and looks up the corresponding spatial embeddings.
        """
        try:
            # Get the maximum number of position embeddings from the layer's configuration.
            # This value is used to scale the float coordinates into integer bins.
            max_len = self.config.max_2d_position_embeddings

            # Scale the float coordinates (0.0 to 1.0) to be integer indices (0 to max_len-1)
            # and cast them to the required 'long' data type for the embedding lookup.
            # We also clamp the values between 0 and 1 to prevent out-of-bounds errors.
            left_indices = (bbox[:, :, 0].clamp(0, 1) * (max_len - 1)).long()
            upper_indices = (bbox[:, :, 1].clamp(0, 1) * (max_len - 1)).long()
            right_indices = (bbox[:, :, 2].clamp(0, 1) * (max_len - 1)).long()
            lower_indices = (bbox[:, :, 3].clamp(0, 1) * (max_len - 1)).long()

            # Perform the embedding lookup using the new integer indices.
            left_position_embeddings = self.x_position_embeddings(left_indices)
            upper_position_embeddings = self.y_position_embeddings(upper_indices)
            right_position_embeddings = self.x_position_embeddings(right_indices)
            lower_position_embeddings = self.y_position_embeddings(lower_indices)

            embeddings = (
                    left_position_embeddings
                    + upper_position_embeddings
                    + right_position_embeddings
                    + lower_position_embeddings
            )

            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            embeddings = self.spatial_emb_matcher(embeddings)
            return embeddings
            
        except IndexError as e:
            raise IndexError(f"Error during spatial embedding. It's possible the bounding box coordinates are out of the [0, 1] range. Original error: {e}")

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class VisualEmbeddings(nn.Module):

    def __init__(self, config, finetune=False):
        super(VisualEmbeddings, self).__init__()

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(config.visual_module_config['model_weights'])
        self.image_model = AutoModel.from_pretrained(config.visual_module_config['model_weights'])
        self.visual_emb_matcher = MLP(self.image_model.config.hidden_size, 0, self.image_model.config.hidden_size, 1)

        if not finetune:
            self.freeze()

    def freeze(self):
        for p in self.image_model.parameters():
            p.requires_grad = False

    def get_visual_boxes(self, num_pages=1, scale=1):
        boxes = torch.tensor([[0, 0, 1, 1]] + [[x / 14, y / 14, (x + 1) / 14, (y + 1) / 14] for y in range(0, 14) for x in range(0, 14)], dtype=torch.float32)
        boxes = boxes.unsqueeze(dim=0).expand([num_pages, -1, -1])
        boxes = boxes * scale
        return boxes

    def forward(self, images, page_idx_mask=None):
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        vis_embeddings = self.image_model(inputs.pixel_values.to(self.image_model.device))
        vis_embeddings = vis_embeddings.last_hidden_state  # BS; 14x14+CLS (197); 768 (hidden size)
        vis_embeddings = self.visual_emb_matcher(vis_embeddings)

        if page_idx_mask is not None:
            vis_attention_mask = torch.zeros(vis_embeddings.shape[:2], dtype=torch.long).to(self.image_model.device)
            vis_attention_mask[page_idx_mask] = 1
        else:
            vis_attention_mask = torch.ones(vis_embeddings.shape[:2], dtype=torch.long).to(self.image_model.device)

        return vis_embeddings, vis_attention_mask

class RetrievalModule(nn.Module):

    def __init__(self, config):
        super(RetrievalModule, self).__init__()

        self.page_retrieval = nn.Linear(config.max_doc_pages * config.page_tokens * config.hidden_size, config.max_doc_pages)
        # TODO Check if BinaryCrossEntropy allows to extend to longer sequences.

        if config.page_retrieval_config['loss'].lower() in ['ce', 'crossentropy', 'crossentropyloss']:
            self.retrieval_criterion = nn.CrossEntropyLoss()

        self.retrieval_loss_weight = config.page_retrieval_config['loss_weight']

    def forward(self, document_embeddings, answer_page_idx):
        document_embeddings = document_embeddings.view([len(document_embeddings), -1])
        # document_embeddings = F.pad(document_embeddings, (0, self.page_retrieval.in_features-document_embeddings.shape[-1]), "constant", 0)  # In case is the last batch

        try:
            ret_logits = self.page_retrieval(document_embeddings)  # 10*2*512

        except:
            pad_document_embeddings = torch.zeros([len(document_embeddings), self.page_retrieval.in_features], dtype=document_embeddings.dtype, device=document_embeddings.device)
            pad_document_embeddings[:, :document_embeddings.shape[-1]] = document_embeddings
            ret_logits = self.page_retrieval(pad_document_embeddings.to())  # 10*2*512

        ret_loss = self.retrieval_criterion(ret_logits, answer_page_idx) * self.retrieval_loss_weight if answer_page_idx is not None else None

        return ret_loss, ret_logits

def to_lora_model(model, config):
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        target_modules=config.targets,
        # modules_to_save=config.to_save,
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
