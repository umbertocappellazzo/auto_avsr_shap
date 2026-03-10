#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 10:45:20 2026

@author: umbertocappellazzo
"""

import torch
import torchaudio
from cosine import WarmupCosineScheduler
from datamodule.transforms import TextTransform

from pytorch_lightning import LightningModule
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_conformer_av import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.scorers.ctc import CTCPrefixScorer

import shap
import os
import numpy as np
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.mask import target_mask


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())

class ModelModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.backbone_args = self.cfg.model.audiovisual_backbone

        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list
        self.model = E2E(len(self.token_list), self.backbone_args)

        # -- initialise
        if self.cfg.pretrained_model_path:
            ckpt = torch.load(self.cfg.pretrained_model_path, map_location=lambda storage, loc: storage)
            if self.cfg.transfer_frontend:
                tmp_ckpt = {k: v for k, v in ckpt["model_state_dict"].items() if k.startswith("trunk.") or k.startswith("frontend3D.")}
                self.model.encoder.frontend.load_state_dict(tmp_ckpt)
            elif self.cfg.transfer_encoder:
                tmp_ckpt = {k.replace("encoder.", ""): v for k, v in ckpt.items() if k.startswith("encoder.")}
                self.model.encoder.load_state_dict(tmp_ckpt, strict=True)
            else:
                self.model.load_state_dict(ckpt)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([{"name": "model", "params": self.model.parameters(), "lr": self.cfg.optimizer.lr}], weight_decay=self.cfg.optimizer.weight_decay, betas=(0.9, 0.98))
        scheduler = WarmupCosineScheduler(optimizer, self.cfg.optimizer.warmup_epochs, self.cfg.trainer.max_epochs, len(self.trainer.datamodule.train_dataloader()))
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def forward(self, video, audio):
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)
        video_feat, _ = self.model.encoder(video.unsqueeze(0).to(self.device), None)
        audio_feat, _ = self.model.aux_encoder(audio.unsqueeze(0).to(self.device), None)
        audiovisual_feat = self.model.fusion(torch.cat((video_feat, audio_feat), dim=-1))

        audiovisual_feat = audiovisual_feat.squeeze(0)

        nbest_hyps = self.beam_search(audiovisual_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        return predicted
    
    
    
    def forward_shap_autoavsr(self, sample, nsamples=2000, shap_alg="kernel"):
        """
        Compute SHAP values for Auto-AVSR (decoder-only).
        
        """
        
        device = self.device
        
        # 1) Encode audio and video
        video = sample["video"].unsqueeze(0).to(device)
        audio = sample["audio"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            video_feat, _ = self.model.encoder(video, None)
            audio_feat, _ = self.model.aux_encoder(audio, None)
        
        B, T_v, D_v = video_feat.shape
        B, T_a, D_a = audio_feat.shape
        
        assert T_v == T_a, f"Video ({T_v}) and audio ({T_a}) must be temporally aligned!"
        T = T_v
        
        #If we wanna check what happens when we zero audio or video features.
        
        #video_feat[:, :, :] = 0  # Zero ALL audio features
        
        # Define feature order (must match concatenation order)
        N_v = T  # Video timesteps (first in concatenation)
        N_a = T  # Audio timesteps (second in concatenation)
        p = N_v + N_a  # Total features
        
        # Store for wrapper
        self.video_feat_full = video_feat
        self.audio_feat_full = audio_feat
        
        # 2) Generate baseline using decoder-only beam search
        concat_full = torch.cat([video_feat, audio_feat], dim=-1)
        memory_full = self.model.fusion(concat_full)
        
        if self.model.proj_decoder:
            memory_full = self.model.proj_decoder(memory_full)
        
        with torch.no_grad():
            beam_search = get_beam_search_decoder(
                self.model,
                self.token_list,
                ctc_weight=0.0,
                beam_size=40
            )
            
            nbest_hyps = beam_search(memory_full.squeeze(0))
            nbest_hyps = [h.asdict() for h in nbest_hyps[:min(len(nbest_hyps), 1)]]
            baseline_token_ids = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        
        self.baseline_token_ids = baseline_token_ids
        
        # Safety check
        if len(baseline_token_ids) == 0:
            raise ValueError("Baseline generation failed: no tokens generated")
        
        # 3) SHAP computation
        background = np.zeros((1, p), dtype=np.float32)
        x_explain = np.ones((1, p), dtype=np.float32)
        
        def shap_model(masks):
            return self.shap_wrapper_autoavsr(masks)
        
        if shap_alg == "sampling":
            explainer = shap.SamplingExplainer(
                model=shap_model,
                data=background
            )
            shap_values = explainer.shap_values(x_explain, nsamples=nsamples)
            
        elif shap_alg == "permutation":
            from shap.maskers import Independent
            masker = Independent(background, max_samples=100)
            
            explainer = shap.PermutationExplainer(
                model=shap_model,
                masker=masker,
                algorithm='auto'
            )
            shap_obj = explainer(x_explain, max_evals=nsamples, silent=True)
            shap_values = shap_obj.values
        
        else:
            raise ValueError(f"Unknown SHAP algorithm: {shap_alg}")
        
        # Process SHAP output
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[0]
        
        vals = shap_values  # (p, T_out)
        
        # 4) Compute contributions
       
        mm_raw_abs = np.sum(np.abs(vals), axis=1)
        mm_video_abs = mm_raw_abs[:N_v].sum()
        mm_audio_abs = mm_raw_abs[N_v:].sum()
        total_abs = mm_audio_abs + mm_video_abs
        
        audio_pct_abs = mm_audio_abs / total_abs
        video_pct_abs = mm_video_abs / total_abs
        
        # Print results
        print(f"Audio contribution (absolute): {audio_pct_abs*100:.2f}%")
        print(f"Video contribution (absolute): {video_pct_abs*100:.2f}%")
        
        return audio_pct_abs, video_pct_abs, \
               T_a, vals

    def shap_wrapper_autoavsr(self, masks):
        """
        SHAP wrapper for evaluating coalitions.
        
        Args:
            masks: (n_coalitions, p) where p = N_v + N_a
                   First N_v elements: video timestep masks
                   Last N_a elements: audio timestep masks
        
        Returns:
            results: (n_coalitions, T_out) logits for baseline tokens
        """
        if masks.ndim == 1:
            masks = masks.reshape(1, -1)
        
        n_coalitions = masks.shape[0]
        device = self.device
        
        T = self.video_feat_full.shape[1]
        N_v = T  # Video timesteps (first in mask)
        
        results = []
        
        for i in range(n_coalitions):
            mask = masks[i]
            
            # Split mask (order: video first, audio second)
            mask_video = mask[:N_v]
            mask_audio = mask[N_v:]
            
            # Clone and mask features
            audio_feat_masked = self.audio_feat_full.clone()
            video_feat_masked = self.video_feat_full.clone()
            
            for t in range(T):
                if mask_video[t] == 0:
                    video_feat_masked[:, t, :] = 0
                if mask_audio[t] == 0:
                    audio_feat_masked[:, t, :] = 0
            
            # Concatenate (video first, audio second)
            concat_masked = torch.cat([video_feat_masked, audio_feat_masked], dim=-1)
            memory_masked = self.model.fusion(concat_masked)
            
            if self.model.proj_decoder:
                memory_masked = self.model.proj_decoder(memory_masked)
            
            # Teacher forcing with baseline tokens
            with torch.no_grad():
                ys_in_pad, ys_out_pad = add_sos_eos(
                    self.baseline_token_ids.unsqueeze(0).to(device),
                    self.model.sos,
                    self.model.eos,
                    self.model.ignore_id
                )
                
                ys_mask = target_mask(ys_in_pad, self.model.ignore_id)
                
                pred_pad, _ = self.model.decoder(
                    ys_in_pad,
                    ys_mask,
                    memory_masked,
                    None
                )
                
                # Extract logits for baseline tokens
                logit_vec = []
                for t, token_id in enumerate(self.baseline_token_ids):
                    if t < pred_pad.shape[1]:
                        logit_vec.append(pred_pad[0, t, token_id].item())
                
                results.append(logit_vec)
        
        return np.array(results)
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="val")

    def test_step(self, sample, sample_idx):
        
        audio_shap_abs_current, video_shap_abs_current, num_audio_tokens, shapley_values = self.forward_shap_autoavsr(
                                                                                    sample, 
                                                                                    nsamples=self.cfg.decode.num_samples_shap,
                                                                                    shap_alg=self.cfg.decode.shap_alg
                                                                                    )
        self.audio_shap_abs.append(audio_shap_abs_current)
        self.video_shap_abs.append(video_shap_abs_current)
        self.num_audio_tokens.append(num_audio_tokens)
        self.shapley_values.append(shapley_values)
        
        self.log("sample-audio-ABS-SHAP", audio_shap_abs_current, on_step=True, on_epoch=False, prog_bar=False)
        self.log("sample-video-ABS-SHAP", video_shap_abs_current, on_step=True, on_epoch=False, prog_bar=False)
        self.log("sample-num-audio-tokens", num_audio_tokens, on_step=True, on_epoch=False, prog_bar=False)

    def _step(self, batch, batch_idx, step_type):
        loss, loss_ctc, loss_att, acc = self.model(batch["videos"], batch["audios"], batch["video_lengths"], batch["audio_lengths"], batch["targets"])
        batch_size = len(batch["videos"])

        if step_type == "train":
            self.log("loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("loss_ctc", loss_ctc, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("loss_att", loss_att, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("decoder_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size)
        else:
            self.log("loss_val", loss, batch_size=batch_size)
            self.log("loss_ctc_val", loss_ctc, batch_size=batch_size)
            self.log("loss_att_val", loss_att, batch_size=batch_size)
            self.log("decoder_acc_val", acc, batch_size=batch_size)

        if step_type == "train":
            self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss

    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.loaders.batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)
        return super().on_train_epoch_start()

    def on_test_epoch_start(self):
        
        self.audio_shap_abs = []
        self.video_shap_abs = []
        self.num_audio_tokens = []
        self.shapley_values = []
       
        self.output_file = os.path.join(
           self.cfg.decode.output_path,
           self.cfg.decode.exp_name
           
        )
        print("Output dir: ", self.output_file)
        
        self.text_transform = TextTransform()
        
        # We do not decode using CTC. I observed little variations by dropping it.
        self.beam_search = get_beam_search_decoder(self.model, self.token_list,  ctc_weight=0.)

    def on_test_epoch_end(self):
        
        overall_audio_abs = np.mean(self.audio_shap_abs)
        overall_video_abs = np.mean(self.video_shap_abs)
        overall_num_audio_tokens = np.mean(self.num_audio_tokens)
        
        std_overall_audio_abs = np.std(self.audio_shap_abs)
        std_overall_video_abs = np.std(self.video_shap_abs)

        self.log("audio-ABS-SHAP", overall_audio_abs)
        self.log("video-ABS-SHAP", overall_video_abs)
        self.log("STD_audio-ABS-SHAP", std_overall_audio_abs)
        self.log("STD_video-ABS-SHAP", std_overall_video_abs)
        self.log("num-audio-tokens", overall_num_audio_tokens)
    
        print("Global Audio-ABS-SHAP :", overall_audio_abs * 100, "%")
        print("Global Video-ABS-SHAP :", overall_video_abs * 100, "%")
        
        np.savez_compressed(
                self.output_file,
                # Aggregated metrics
                audio_abs=np.array(self.audio_shap_abs),
                video_abs=np.array(self.video_shap_abs),
                audio_pos=np.array(self.audio_shap_pos),
                video_pos=np.array(self.video_shap_pos),
                audio_neg=np.array(self.audio_shap_neg),
                video_neg=np.array(self.video_shap_neg),
                num_audio_tokens=np.array(self.num_audio_tokens),
                # Raw SHAP values (ragged array - stored as object array)
                shap_values=np.array(self.shapley_values, dtype=object),
            )


def get_beam_search_decoder(model, token_list, ctc_weight=0.1, beam_size=40):
    scorers = {
        "decoder": model.decoder,
        "ctc": CTCPrefixScorer(model.ctc, model.eos),
        "length_bonus": LengthBonus(len(token_list)),
        "lm": None
    }

    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": 0.0,
        "length_bonus": 0.0,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )
