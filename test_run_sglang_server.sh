# install numactl
apt-get update && apt-get install -y numactl

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# install sglang
uv pip install "sglang[all]" --system
# our version of vllms does not work with transformers 4.54.0
# uv pip install "transformers<4.54.0"

# manually run sglang server and test
# python -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct
# python minimal_areal_chat_test.py

# or e2e run with
# composer -n 2 test_single_controller_sglang.py

