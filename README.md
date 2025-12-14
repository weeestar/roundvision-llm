# Get project
```sh
git clone https://github.com/weeestar/roundvision-llm.git
```

# Install project
```sh
python3 -m venv roundvision-llm
source roundvision-llm/bin/activate
pip install -r requirements.txt
```

# Install HF
```sh
curl -LsSf https://hf.co/cli/install.sh | bash
hf auth login
```

# Models with HF
```sh
hf download nvidia/Mistral-7B-Instruct-v0.3-ONNX-INT4 --local-dir ~/llm-models/mistral‑onnx‑int4
```

# Models without HF
```sh
mkdir ~/llm-models && cd ~/llm-models
git clone --single-branch --branch main https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
```
