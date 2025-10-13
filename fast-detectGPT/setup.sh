# clean cache
conda clean --all -y
conda remove -n fastdetectGPT --all -y
conda create -n fastdetectGPT python=3.11 -y
source activate fastdetectGPT
conda clean --all -y
pip install --upgrade pip
pip install -r requirements.txt
pip install -U "datasets>=2.16.0,<3" "fsspec>=2023.10.0" "pyarrow>=14,<17"
pip install python-dotenv
pip install huggingface_hub
pip install tiktoken
