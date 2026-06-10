import json
import torch

from loguru import logger
from huggingface_hub import hf_hub_download
from transformers import AlbertConfig

from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder

from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel


class KConfigJson(BaseModel):
    vocab: Any
    style_dim: Any
    n_layer: Any
    hidden_dim: Any
    n_token: Any
    plbert: Dict
    dropout: Any
    max_dur: Any
    text_encoder_kernel_size: Any
    n_mels: Any
    istftnet: Dict


class PfModelConfig(KConfigJson):
    repo_id: str = 'hexgrad/Kokoro-82M'
    model_path: str | None = None
    disable_complex: bool = False
    map_location: str = "cpu"
    weights_only: bool = True
    model: str | None = None


class ModelBuilderOptions(BaseModel):
    repo_id: Optional[str] = None
    config: Union[Dict, str, None] = None
    model: Optional[str] = None
    disable_complex: bool = False

    def __init__(self, **kargs: Any):
        'hexgrad/Kokoro-82M'
        super().__init__(**kargs)
        if self.repo_id is None:
            self.repo_id = 'hexgrad/Kokoro-82M'
            print(
                    f"WARNING: Defaulting repo_id to {self.repo_id}."
                    f" Pass repo_id='{self.repo_id}' to suppress this warning."
                    )


def k_model_builder(**kwargs):
    opt = ModelBuilderOptions(**kwargs)
    assert opt.repo_id

    if not isinstance(opt.config, dict):
        if not opt.config:
            logger.debug("No config provided, downloading from HF")
            config_path = hf_hub_download(repo_id=opt.repo_id,
                                          filename='config.json'
                                          )
        else:
            config_path = opt.config
        with open(config_path, 'r', encoding='utf-8') as r:
            config = json.load(r)
            logger.debug(f"Loaded config: {config}")
    else:
        config = opt.config

    pfmodel_config = PfModelConfig(
            repo_id=opt.repo_id,
            model_path=opt.model,
            disable_complex=opt.disable_complex,
            **config
            )
    model_instance = KModel(pfmodel_config)
    return model_instance


class KModel(torch.nn.Module):
    '''
    KModel is a torch.nn.Module with 2 main responsibilities:
    1. Init weights, downloading config.json + model.pth from HF if needed
    2. forward(phonemes: str, ref_s: FloatTensor) -> (audio: FloatTensor)

    You likely only need one KModel instance, and it can be reused across
    multiple KPipelines to avoid redundant memory allocation.

    Unlike KPipeline, KModel is language-blind.

    KModel stores self.vocab and thus knows how to map phonemes -> input_ids,
    so there is no need to repeatedly download config.json outside of KModel.
    '''

    MODEL_NAMES = {
        'hexgrad/Kokoro-82M': 'kokoro-v1_0.pth',
        'hexgrad/Kokoro-82M-v1.1-zh': 'kokoro-v1_1-zh.pth',
    }

    def __init__(
        self,
        config: PfModelConfig,
    ):

        super().__init__()

        self.vocab = config.vocab
        self.bert = CustomAlbert(
                AlbertConfig(
                    vocab_size=config.n_token,
                    **config.plbert)
                )

        self.bert_encoder = torch.nn.Linear(
                self.bert.config.hidden_size,
                config.hidden_dim
                )

        self.context_length = self.bert.config.max_position_embeddings

        self.predictor = ProsodyPredictor(
                style_dim=config.style_dim,
                d_hid=config.hidden_dim,
                nlayers=config.n_layer,
                max_dur=config.max_dur,
                dropout=config.dropout
                )

        self.text_encoder = TextEncoder(
                channels=config.hidden_dim,
                kernel_size=config.text_encoder_kernel_size,
                depth=config.n_layer,
                n_symbols=config.n_token
                )

        self.decoder = Decoder(
                dim_in=config.hidden_dim,
                style_dim=config.style_dim,
                dim_out=config.n_mels,
                disable_complex=config.disable_complex,
                **config.istftnet
                )

        self.repo_id = config.repo_id
        if not config.model_path:
            self.download_model()
        else:
            self.load_model(config.model_path)


    def download_model(self):
        model_path = hf_hub_download(repo_id=self.repo_id,
                                     filename=KModel.MODEL_NAMES[self.repo_id]
                                     )
        self.load_model(model_path)

    def load_model(self, model_path: str):
        loaded = torch.load(model_path, map_location='cpu', weights_only=True)

        for key, state_dict in loaded.items():
            assert hasattr(self, key), key
            try:
                getattr(self, key).load_state_dict(state_dict)
            except:
                logger.debug(f"Did not load {key} from state_dict")
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                getattr(self, key).load_state_dict(state_dict, strict=False)

    @property
    def device(self):
        return self.bert.device

    @dataclass
    class Output:
        audio: torch.FloatTensor
        pred_dur: Optional[torch.LongTensor] = None

    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:

        input_lengths = torch.full(
            (input_ids.shape[0],), 
            input_ids.shape[-1], 
            device=input_ids.device,
            dtype=torch.long
        )

        text_mask = (
                torch.gt(
                    ( 1 + (
                        torch.arange(float(input_lengths.max()))
                        .unsqueeze(0)
                        .expand(input_lengths.shape[0], -1)
                        .type_as(input_lengths)
                        ) 
                     )
                    ,
                    input_lengths.unsqueeze(1)).to(self.device)
                )

        ref_s_tail = ref_s[:, 128:]
        ref_s_head = ref_s[:, :128]
        d = self.predictor.text_encoder(
                (
                    self.bert_encoder(
                        self.bert(
                            input_ids,
                            attention_mask=(
                                ~ text_mask
                                ).int()
                            )
                        ).transpose(-1, -2)
                    )
                ,
                ref_s_tail,
                input_lengths,
                text_mask
                ,
                )

        prediction_duration= (
                torch.round(
                    torch.sigmoid(
                        self.predictor.duration_proj(
                            # Project Long Shor Term Memory
                            self.predictor.lstm(d)[0]
                            )
                        )
                            .sum(dim=-1)
                            / speed
                    )
                .clamp(min=1)
                .long()
                .squeeze()
                )

        indices = torch.repeat_interleave(
                torch.arange(
                    input_ids.shape[1],
                    device=self.device
                    )
                ,
                prediction_duration
                )

        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=self.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)

        F0_pred, N_pred = self.predictor.F0Ntrain(
                d.transpose(-1, -2)
                @
                pred_aln_trg
                ,
                ref_s_tail
                )

        audio = self.decoder(
                (
                    self.text_encoder(
                        input_ids,
                        input_lengths,
                        text_mask
                        )
                    @
                    pred_aln_trg
                )
                ,
                F0_pred,
                N_pred,
                ref_s_head
                ,
                ).squeeze()

        return audio, prediction_duration

    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        return_output: bool = False
    ) -> Union['KModel.Output', torch.FloatTensor]:

        input_ids = [self.vocab[p] for p in phonemes if self.vocab.get(p) is not None]
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)
        ref_s = ref_s.to(self.device)

        logger.debug(f"phonemes: {phonemes} -> input_ids: {input_ids}")
        assert len(input_ids) + 2 <= self.context_length, "ids > (context - 2)"
        audio, pred_dur = self.forward_with_tokens(input_ids, ref_s, speed)
        audio = audio.squeeze().cpu()

        if pred_dur is not None:
            pred_dur = pred_dur.cpu()
        logger.debug(f"pred_dur: {pred_dur}")

        if return_output:
            return self.Output(audio=audio, pred_dur=pred_dur)
        return audio

class KModelForONNX(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel

    def forward(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        waveform, duration = self.kmodel.forward_with_tokens(input_ids, ref_s, speed)
        return waveform, duration
