conda create -y --name ppggptft python=3.11
conda activate ppggptft
conda install -y cmake ninja packaging ipython pybind11 ccache scikit-learn pyyaml
conda install -y -c conda-forge htop
pip install natsort umap-learn tables scipy openpyxl PyWavelets neurokit2 statsmodels Bottleneck zarr wfdb gpustat kymatio peft dotmap tqdm matplotlib h5py omegaconf dacite seaborn pydantic
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install einops sympy==1.13.1 fsspec pytest
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
# MAX_JOBS=1 pip -v install flash-attn --no-build-isolation
pip install fire safetensors wandb rotary_embedding_torch==0.6.4 accelerate
