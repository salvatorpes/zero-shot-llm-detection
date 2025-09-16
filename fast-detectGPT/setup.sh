conda remove -n fastdetectGPT --all -y
conda create -n fastdetectGPT python=3.9 -y
conda activate fastdetectGPT
pip install tokenizers
pip install -r requirements.txt
