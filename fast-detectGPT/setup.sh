conda remove -n fastdetectGPT --all -y
conda create -n fastdetectGPT python=3.8 -y
conda activate fastdetectGPT
pip install --upgrade pip
pip install -r requirements.txt
pip install huggingface_hub
huggingface-cli login
pip install -U "datasets>=2.16.0,<3" "fsspec>=2023.10.0" "huggingface_hub>=0.20.0" "pyarrow>=14,<17"