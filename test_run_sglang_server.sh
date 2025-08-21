# install numactl
apt install numactl

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# install sglang
uv venv sglang --system-site-packages
source sglang/bin/activate
uv pip install "sglang[all]"

# manually run sglang server and test
# python -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct
# python minimal_areal_chat_test.py

# or e2e run with
# python test_single_controller_sglang.py

