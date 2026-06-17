import inspect
import json
import os
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME

try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
except ImportError:  # pragma: no cover - optional dependency
    safe_load_file = None
    safe_save_file = None


TISSUE_PROMPT_CONFIG_NAME = "tissue_prompt_config.json"
TOKEN_TASK_LEVEL = "token"
SEQUENCE_TASK_LEVEL = "sequence"
REGRESSION_PROBLEM_TYPE = "regression"
SINGLE_LABEL_CLASSIFICATION_PROBLEM_TYPE = "single_label_classification"
MULTI_LABEL_CLASSIFICATION_PROBLEM_TYPE = "multi_label_classification"


def _get_hidden_size(config) -> int:
    for attr_name in ("hidden_size", "d_model", "n_embd", "dim"):
        value = getattr(config, attr_name, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer hidden size from the base model config.")


def _get_classifier_dropout(task_model: nn.Module) -> float:
    config = getattr(task_model, "config", None)
    if config is not None:
        for attr_name in ("classifier_dropout", "hidden_dropout_prob", "dropout"):
            value = getattr(config, attr_name, None)
            if value is not None:
                return float(value)
    dropout_module = getattr(task_model, "dropout", None)
    if isinstance(dropout_module, nn.Dropout):
        return float(dropout_module.p)
    return 0.1


class TissuePromptModel(nn.Module):
    """
    Wrapper around an HF token/sequence classification model that injects
    learned tissue-conditioned virtual tokens after position 0 ([CLS]).
    """

    def __init__(
        self,
        task_model: nn.Module,
        *,
        task_level: str,
        num_tissue_types: int,
        num_virtual_tokens: int,
        tissue_embedding_dim: Optional[int] = None,
        projector_hidden_dim: Optional[int] = None,
        problem_type: Optional[str] = None,
        ignore_index: int = -100,
        token_labels_include_cls: bool = True,
        backbone_name_or_path: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        super().__init__()
        if task_level not in {TOKEN_TASK_LEVEL, SEQUENCE_TASK_LEVEL}:
            raise ValueError(f"Unsupported task_level: {task_level}")
        if num_virtual_tokens < 1:
            raise ValueError("num_virtual_tokens must be >= 1.")
        if num_tissue_types < 1:
            raise ValueError("num_tissue_types must be >= 1.")

        self.task_model = task_model
        self.config = task_model.config
        self.task_level = task_level
        self.num_virtual_tokens = int(num_virtual_tokens)
        self.num_tissue_types = int(num_tissue_types)
        self.ignore_index = int(ignore_index)
        self.token_labels_include_cls = bool(token_labels_include_cls)
        self.backbone_name_or_path = backbone_name_or_path
        self.trust_remote_code = bool(trust_remote_code)

        hidden_size = _get_hidden_size(self.config)
        tissue_embedding_dim = int(tissue_embedding_dim or hidden_size)
        projector_hidden_dim = int(projector_hidden_dim or hidden_size)

        self.tissue_embeddings = nn.Embedding(self.num_tissue_types, tissue_embedding_dim)
        self.tissue_projector = nn.Sequential(
            nn.Linear(tissue_embedding_dim, projector_hidden_dim),
            nn.GELU(),
            nn.Linear(projector_hidden_dim, self.num_virtual_tokens * hidden_size),
        )
        self.prompt_dropout = nn.Dropout(_get_classifier_dropout(task_model))

        if problem_type is None:
            if getattr(self.config, "num_labels", 1) == 1:
                problem_type = REGRESSION_PROBLEM_TYPE
            else:
                problem_type = SINGLE_LABEL_CLASSIFICATION_PROBLEM_TYPE
        self.problem_type = problem_type
        self.config.problem_type = problem_type

    @property
    def base_model(self) -> nn.Module:
        return self.task_model.base_model

    def get_input_embeddings(self) -> nn.Module:
        return self.base_model.get_input_embeddings()

    def _get_classifier(self) -> nn.Module:
        for module_name in ("classifier", "score"):
            module = getattr(self.task_model, module_name, None)
            if module is not None:
                return module
        raise AttributeError("Could not find a token/sequence classification head on the wrapped model.")

    def _get_head_dropout(self) -> nn.Module:
        dropout_module = getattr(self.task_model, "dropout", None)
        if dropout_module is None:
            return nn.Identity()
        return dropout_module

    def _get_prompt_config(self) -> Dict[str, Any]:
        return {
            "task_level": self.task_level,
            "num_tissue_types": self.num_tissue_types,
            "num_virtual_tokens": self.num_virtual_tokens,
            "tissue_embedding_dim": int(self.tissue_embeddings.embedding_dim),
            "projector_hidden_dim": int(self.tissue_projector[0].out_features),
            "problem_type": self.problem_type,
            "ignore_index": self.ignore_index,
            "token_labels_include_cls": self.token_labels_include_cls,
            "backbone_name_or_path": self.backbone_name_or_path,
            "trust_remote_code": self.trust_remote_code,
        }

    @classmethod
    def from_backbone_pretrained(
        cls,
        backbone_name_or_path: str,
        *,
        num_labels: int,
        task_level: str,
        num_tissue_types: int,
        num_virtual_tokens: int,
        tissue_embedding_dim: Optional[int] = None,
        projector_hidden_dim: Optional[int] = None,
        problem_type: Optional[str] = None,
        ignore_index: int = -100,
        token_labels_include_cls: bool = True,
        trust_remote_code: bool = True,
        load_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "TissuePromptModel":
        load_kwargs = dict(load_kwargs or {})
        if task_level == TOKEN_TASK_LEVEL:
            task_model = AutoModelForTokenClassification.from_pretrained(
                backbone_name_or_path,
                num_labels=num_labels,
                trust_remote_code=trust_remote_code,
                **load_kwargs,
            )
        else:
            task_model = AutoModelForSequenceClassification.from_pretrained(
                backbone_name_or_path,
                num_labels=num_labels,
                trust_remote_code=trust_remote_code,
                **load_kwargs,
            )
        return cls(
            task_model,
            task_level=task_level,
            num_tissue_types=num_tissue_types,
            num_virtual_tokens=num_virtual_tokens,
            tissue_embedding_dim=tissue_embedding_dim,
            projector_hidden_dim=projector_hidden_dim,
            problem_type=problem_type,
            ignore_index=ignore_index,
            token_labels_include_cls=token_labels_include_cls,
            backbone_name_or_path=backbone_name_or_path,
            trust_remote_code=trust_remote_code,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **override_kwargs) -> "TissuePromptModel":
        prompt_config_path = os.path.join(pretrained_model_name_or_path, TISSUE_PROMPT_CONFIG_NAME)
        if not os.path.exists(prompt_config_path):
            raise FileNotFoundError(f"Missing {TISSUE_PROMPT_CONFIG_NAME} under {pretrained_model_name_or_path}")

        with open(prompt_config_path, "r", encoding="utf-8") as handle:
            prompt_config = json.load(handle)
        prompt_config.update(override_kwargs)

        trust_remote_code = prompt_config.get("trust_remote_code", True)
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
        task_level = prompt_config["task_level"]
        if task_level == TOKEN_TASK_LEVEL:
            task_model = AutoModelForTokenClassification.from_config(config, trust_remote_code=trust_remote_code)
        else:
            task_model = AutoModelForSequenceClassification.from_config(config, trust_remote_code=trust_remote_code)

        model = cls(
            task_model,
            task_level=task_level,
            num_tissue_types=prompt_config["num_tissue_types"],
            num_virtual_tokens=prompt_config["num_virtual_tokens"],
            tissue_embedding_dim=prompt_config.get("tissue_embedding_dim"),
            projector_hidden_dim=prompt_config.get("projector_hidden_dim"),
            problem_type=prompt_config.get("problem_type"),
            ignore_index=prompt_config.get("ignore_index", -100),
            token_labels_include_cls=prompt_config.get("token_labels_include_cls", True),
            backbone_name_or_path=prompt_config.get("backbone_name_or_path"),
            trust_remote_code=trust_remote_code,
        )

        safe_weights_path = os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_NAME)
        torch_weights_path = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        if os.path.exists(safe_weights_path):
            if safe_load_file is None:
                raise ImportError("safetensors is required to load a safetensors checkpoint.")
            state_dict = safe_load_file(safe_weights_path)
        elif os.path.exists(torch_weights_path):
            state_dict = torch.load(torch_weights_path, map_location="cpu", weights_only=False)
        else:
            raise FileNotFoundError(
                f"Expected {SAFE_WEIGHTS_NAME} or {WEIGHTS_NAME} under {pretrained_model_name_or_path}"
            )

        model.load_state_dict(state_dict, strict=True)
        return model

    def save_pretrained(
        self,
        save_directory: str,
        *,
        safe_serialization: bool = True,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
        **_,
    ) -> None:
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        with open(os.path.join(save_directory, TISSUE_PROMPT_CONFIG_NAME), "w", encoding="utf-8") as handle:
            json.dump(self._get_prompt_config(), handle, indent=2, sort_keys=True)

        state_dict = state_dict or self.state_dict()
        if safe_serialization and safe_save_file is not None:
            safe_save_file(state_dict, os.path.join(save_directory, SAFE_WEIGHTS_NAME))
            return

        torch.save(state_dict, os.path.join(save_directory, WEIGHTS_NAME))

    def _reshape_tissue_prompts(self, tissue_ids: torch.LongTensor) -> torch.Tensor:
        tissue_embeds = self.tissue_embeddings(tissue_ids)
        prompt_flat = self.tissue_projector(tissue_embeds)
        hidden_size = self.get_input_embeddings().embedding_dim
        prompt_embeds = prompt_flat.view(-1, self.num_virtual_tokens, hidden_size)
        return self.prompt_dropout(prompt_embeds)

    def _insert_after_cls(self, tensor: torch.Tensor, insert_tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat([tensor[:, :1], insert_tensor, tensor[:, 1:]], dim=1)

    def _extend_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_ids: torch.LongTensor,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        prompt_attention = attention_mask.new_ones((attention_mask.shape[0], self.num_virtual_tokens))
        return self._insert_after_cls(attention_mask, prompt_attention)

    def _extend_input_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        prompt_token_id = getattr(self.config, "pad_token_id", 0)
        if prompt_token_id is None or int(prompt_token_id) < 0:
            prompt_token_id = 0
        prompt_ids = input_ids.new_full((input_ids.shape[0], self.num_virtual_tokens), int(prompt_token_id))
        return self._insert_after_cls(input_ids, prompt_ids)

    def _extend_token_type_ids(self, token_type_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if token_type_ids is None:
            return None
        prompt_token_types = token_type_ids[:, :1].expand(-1, self.num_virtual_tokens)
        return self._insert_after_cls(token_type_ids, prompt_token_types)

    def _extend_position_ids(self, position_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if position_ids is None:
            return None
        cls_positions = position_ids[:, :1]
        prompt_offsets = torch.arange(
            1,
            self.num_virtual_tokens + 1,
            device=position_ids.device,
            dtype=position_ids.dtype,
        ).unsqueeze(0)
        prompt_positions = cls_positions + prompt_offsets
        # TODO: if you want real tokens to keep their original positions,
        # replace this natural shift with custom position_ids for the real tokens.
        shifted_real_positions = position_ids[:, 1:] + self.num_virtual_tokens
        return torch.cat([cls_positions, prompt_positions, shifted_real_positions], dim=1)

    def _filter_backbone_kwargs(self, backbone_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        signature = inspect.signature(self.base_model.forward)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        )
        if accepts_kwargs:
            return backbone_kwargs
        accepted_names = set(signature.parameters.keys())
        return {key: value for key, value in backbone_kwargs.items() if key in accepted_names}

    def _needs_inputs_embeds_workaround(self) -> bool:
        embeddings = getattr(self.base_model, "embeddings", None)
        return embeddings is not None and bool(getattr(embeddings, "token_dropout", False))

    def _forward_backbone_with_inputs_embeds(
        self,
        augmented_input_ids: torch.LongTensor,
        backbone_kwargs: Dict[str, Any],
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        attention_mask = backbone_kwargs.get("attention_mask")
        if attention_mask is None:
            attention_mask = augmented_input_ids.new_ones(augmented_input_ids.shape)

        input_shape = augmented_input_ids.size()
        extended_attention_mask = self.base_model.get_extended_attention_mask(attention_mask, input_shape)

        encoder_hidden_states = backbone_kwargs.get("encoder_hidden_states")
        encoder_attention_mask = backbone_kwargs.get("encoder_attention_mask")
        if self.base_model.config.is_decoder and encoder_hidden_states is not None:
            encoder_hidden_shape = encoder_hidden_states.size()[:-1]
            if encoder_attention_mask is None:
                encoder_attention_mask = attention_mask.new_ones(encoder_hidden_shape)
            encoder_extended_attention_mask = self.base_model.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.base_model.get_head_mask(
            backbone_kwargs.get("head_mask"),
            self.base_model.config.num_hidden_layers,
        )
        past_key_values = backbone_kwargs.get("past_key_values")
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        embedding_output = self.base_model.embeddings(
            input_ids=augmented_input_ids,
            position_ids=backbone_kwargs.get("position_ids"),
            attention_mask=attention_mask,
            inputs_embeds=backbone_kwargs["inputs_embeds"],
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.base_model.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=backbone_kwargs.get("use_cache", False),
            output_attentions=backbone_kwargs.get("output_attentions"),
            output_hidden_states=backbone_kwargs.get("output_hidden_states"),
            return_dict=True,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.base_model.pooler(sequence_output) if self.base_model.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=getattr(encoder_outputs, "past_key_values", None),
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=getattr(encoder_outputs, "cross_attentions", None),
        )

    def _compute_token_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid_mask = labels != self.ignore_index
        if not torch.any(valid_mask):
            return logits.sum() * 0.0

        if self.problem_type == REGRESSION_PROBLEM_TYPE:
            squeezed_logits = logits.squeeze(-1)
            return F.mse_loss(squeezed_logits[valid_mask], labels.float()[valid_mask])

        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask].long()
        return F.cross_entropy(valid_logits, valid_labels)

    def _compute_sequence_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.problem_type == REGRESSION_PROBLEM_TYPE:
            return F.mse_loss(logits.squeeze(-1), labels.float().view_as(logits.squeeze(-1)))
        if self.problem_type == MULTI_LABEL_CLASSIFICATION_PROBLEM_TYPE:
            return F.binary_cross_entropy_with_logits(logits, labels.float())
        return F.cross_entropy(logits.view(-1, self.config.num_labels), labels.view(-1).long())

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        tissue_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if tissue_ids is None:
            raise ValueError("tissue_ids must be provided when using TissuePromptModel.")
        if input_ids is None:
            raise ValueError("input_ids must be provided so the base embedding layer can be reused.")
        if inputs_embeds is not None:
            raise ValueError("Pass input_ids to TissuePromptModel; it constructs inputs_embeds internally.")

        input_embeddings = self.get_input_embeddings()(input_ids)
        prompt_embeddings = self._reshape_tissue_prompts(tissue_ids)

        # Shapes:
        # input_embeddings:    [batch, original_seq_len, hidden]
        # prompt_embeddings:   [batch, K, hidden]
        # augmented_embeds:    [batch, original_seq_len + K, hidden]
        augmented_embeds = self._insert_after_cls(input_embeddings, prompt_embeddings)
        augmented_input_ids = self._extend_input_ids(input_ids)
        augmented_attention_mask = self._extend_attention_mask(attention_mask, input_ids)
        augmented_token_type_ids = self._extend_token_type_ids(token_type_ids)
        augmented_position_ids = self._extend_position_ids(position_ids)

        backbone_kwargs = {
            "inputs_embeds": augmented_embeds,
            "attention_mask": augmented_attention_mask,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }
        if augmented_token_type_ids is not None:
            backbone_kwargs["token_type_ids"] = augmented_token_type_ids
        if augmented_position_ids is not None:
            backbone_kwargs["position_ids"] = augmented_position_ids
        backbone_kwargs.update(kwargs)
        filtered_backbone_kwargs = self._filter_backbone_kwargs(backbone_kwargs)
        if self._needs_inputs_embeds_workaround():
            backbone_outputs = self._forward_backbone_with_inputs_embeds(
                augmented_input_ids=augmented_input_ids,
                backbone_kwargs=filtered_backbone_kwargs,
            )
        else:
            backbone_outputs = self.base_model(**filtered_backbone_kwargs)

        sequence_output = backbone_outputs.last_hidden_state
        classifier = self._get_classifier()
        head_dropout = self._get_head_dropout()
        loss = None

        if self.task_level == TOKEN_TASK_LEVEL:
            # sequence_output layout:
            # [CLS], prompt_1, ..., prompt_K, real_token_1, ..., real_token_N
            real_token_hidden = sequence_output[:, 1 + self.num_virtual_tokens :, :]
            real_token_hidden = head_dropout(real_token_hidden)
            real_token_logits = classifier(real_token_hidden)

            if self.token_labels_include_cls:
                cls_placeholder = real_token_logits.new_zeros(
                    (real_token_logits.shape[0], 1, real_token_logits.shape[-1])
                )
                logits = torch.cat([cls_placeholder, real_token_logits], dim=1)
            else:
                logits = real_token_logits

            if labels is not None:
                loss = self._compute_token_loss(logits, labels)

            if return_dict is False:
                return ((loss, logits) if loss is not None else (logits,))
            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=backbone_outputs.hidden_states,
                attentions=backbone_outputs.attentions,
            )

        cls_hidden = sequence_output[:, 0, :]
        logits = classifier(head_dropout(cls_hidden))
        if labels is not None:
            loss = self._compute_sequence_loss(logits, labels)

        if return_dict is False:
            return ((loss, logits) if loss is not None else (logits,))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=backbone_outputs.hidden_states,
            attentions=backbone_outputs.attentions,
        )
