## Shapley Value computations for Auto-AVSR

This repository contains the code to compute the audio/video SHAP contributions for the Auto-AVSR model. For more details, please refer to our [`paper`](??). 

---

## Requirements

To setup the environment and pre-process the LRS2/LRS3 datasets, please refer to the official [`Auto-AVSR repository`](https://github.com/mpc001/auto_avsr) with all the details. Once this is done, make sure to install the **shap** and **wandb** libraries: ```pip install shap wandb==0.15.12```. In addition to this, download the [`Auto-AVSR checkpoint`](https://drive.google.com/file/d/1mU6MHzXMiq1m6GI-8gqT2zc2bdStuBXu/view) we used in our manuscript. The ckpt we used is the one trained on LRW, LRS2, LRS3, VoxCeleb2, AVSpeech (around 3448 hours).  

## Compute the global A/V-SHAP Contributions.

To compute the A/V-SHAP contributions, run the command as below: 

```Shell
python eval_shap.py data.modality=audiovisual data.dataset.root_dir=[path_to_dataset] decode.wandb_project=[wandb_project] \
data.dataset.test_file=[test_file] pretrained_model_path=[path_to_ckpt] decode.exp_name=[exp_name] \
decode.shap_alg=[shap_alg] decode.num_samples_shap=[num_samples_shap] decode.output_path=[path_to_output_save] decode.snr_target=[snr_target]
```

 <details open>
  <summary><strong>Main Arguments</strong></summary>
    
- `path_to_dataset`: The path where the dataset is located.
- `wandb_project`: Name of the wandb project to track the results.
- `test_file`: The labels test file (e.g., 'lrs2_test_transcript_lengths_seg16s.csv').
- `exp_name`: The experiment name.
- `pretrained_model_path`: The path to the pre-trained ckpt.
- `shap_alg`: The algorithm from the shap library to compute the shapley matrix. Choices: [`sampling`, `permutation`].
- `num_samples_shap`: The number of coalitions to sample. Default: 2000.
- `output_path`: The path to save the SHAP values for further analyses. This folder must be created beforehand!
- `snr_target`: The SNR level of acoustic noise to test on. If no specified, we test in clean conditions.

</details>


---

## 🔖 Citation

If you find our work useful, please cite:

```bibtex
@article{cappellazzo2026ODrSHAPAV,
  title={Dr. SHAP-AV: Decoding Relative Modality Contributions via Shapley Attribution in Audio-Visual Speech Recognition},
  author={Umberto, Cappellazzo and Stavros, Petridis and Maja, Pantic},
  journal={arXiv preprint arXiv:?},
  year={2026}
}
```

---

## 🙏 Acknowledgements

- Our code relies on [auto-avsr](https://github.com/mpc001/auto_avsr)

---

## 📧 Contact

For questions and discussions, please:
- Open an issue on GitHub
- Email: umbertocappellazzo@gmail.com
- Visit our [project page](https://umbertocappellazzo.github.io/Dr-SHAP-AV/) and our [preprint](https://arxiv.org/abs/2511.07253)

---
