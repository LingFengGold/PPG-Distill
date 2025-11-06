<div align="center">
  <!-- <h1><b> TimeDistill </b></h1> -->
  <!-- <h2><b> TimeDistill </b></h2> -->
  <h2><b>[NeurIPS 2025 TS4H] PPG-Distill: Efficient Photoplethysmography Signals Analysis via Foundation Model Distillation</b></h2>
</div>

<p align="center">
<img src="./figures/fig.png" width="100">
</p>

## Quick Run

This guide provides a quick way to get started with training and distilling models using our framework.

### 1. Download Pre-trained PaPaGei Weights

First, you need to download the pre-trained weights for the PaPaGei foundation model.

```bash
# Create a directory for the model weights
mkdir -p papagei-foundation-model/weights

# Download the papagei_s.pt model weights
wget -O papagei-foundation-model/weights/papagei_s.pt https://zenodo.org/records/13983110/files/papagei%5Fs.pt?download=1

(or mannully download the weight from https://github.com/Nokia-Bell-Labs/papagei-foundation-model)
```

### 2. Configure Model Paths

The configuration files for the PaPaGei model expect the weights to be in a specific path. The download command above places the file in the correct location (`papagei-foundation-model/weights/papagei_s.pt`), which matches the default path in the YAML configuration files (`config/models/papagei_config_*.yaml`).

If you choose to store the weights in a different location, you must update the `model_path` in the following files to point to your custom path:
- `config/models/papagei_config_dalia.yaml`
- `config/models/papagei_config_stanfordAF.yaml`

Example of the line to modify in the YAML files:
```yaml
model_path: "path/to/your/papagei_s.pt"
```

### 3. Finetune a Teacher Model

You can finetune a teacher model (e.g., PaPaGei) on a specific dataset. Here is an example command for training on the `dalia` dataset:

```bash
python train.py --model_type papagei --dataset dalia
```

### 4. Distill a Student Model

Once you have a trained teacher model, you can distill its knowledge into a smaller student model. For example, to distill from a PaPaGei teacher to a GPT-1M student on the `dalia` dataset:

```bash
python train_distill.py \
    --teacher_type papagei \
    --student_type gpt_1m \
    --dataset dalia \
    --save_dir './output' \
    --save_dir_student './output_s'
```

This command will:
- Load the pre-trained PaPaGei model from the path specified in `--save_dir`.
- Create a `gpt_1m` student model.
- Run the distillation process.
- Save the distilled student model in the directory specified by `--save_dir_student`.

---
>
> 🧑‍💻 Please let us know if you notice any mistakes or have suggestions!
>
> 🌟 If you find this resource helpful, please consider starring this repository and citing our research:
```
@inproceedings{
    ni2025ppgdistill,
    title={{PPG}-Distill: Efficient Photoplethysmography Signals Analysis via Foundation Model Distillation},
    author={Juntong Ni and Saurabh Kataria and Shengpu Tang and Carl Yang and Xiao Hu and Wei Jin},
    booktitle={NeurIPS 2025 Workshop on Learning from Time Series for Health},
    year={2025},
    url={https://openreview.net/forum?id=OStTScAV5g}
}
```
---