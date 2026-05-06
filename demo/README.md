# ASearcher Demo

This directory contains two separate browser UIs:

- `asearcher_demo.py` + `asearcher_client.html`: interactive demo backed by a vLLM/OpenAI-compatible endpoint.
- `light_trace_server.py` + `light_trace_viewer.html`: local trace viewer for inspecting lightweight training rollout traces.

## Files

- `asearcher_demo.py`: FastAPI backend for the interactive demo.
- `asearcher_client.html`: browser client for the interactive demo.
- `launch_demo.sh`: launcher for the interactive demo backend.
- `light_trace_server.py`: FastAPI server for the trace viewer.
- `light_trace_viewer.html`: browser UI for browsing `*.trace.json` files.
- `launch_trace_viewer.sh`: launcher for the trace viewer.

## Interactive Demo
0. **Installation:**
    ```bash
    pip install openai fastapi uvicorn vllm
    ```

1. **Start the vLLM Server:**
    Before running the demo, you need to have a vLLM server running with the desired model, for example:
    ```bash
    vllm serve path/to/model --host $host --port $port
    ```

2.  **Start the Demo Service:**  
    ```bash
    bash launch_demo.sh \
        http://localhost:8000 \
        Qwen2.5-7B-Instruct \
        8080 \
        0.0.0.0 \
        false
    ```

    Or run the Python entry directly:
    ```bash
    python3 asearcher_demo.py \
        --host $api_host \
        --port $api_port \
        --llm-url [llm_host:vllm_port] \
        --model-name $model_name \
    ```
    You can get our model from [🤗huggingface](https://huggingface.co/collections/inclusionAI/asearcher-6891d8acad5ebc3a1e1fb2d1)

3.  **Open the Client:**
    Open the `asearcher_client.html` file in your web browser to access the user interface.

    You can open it directly from your file system, or serve it via a simple HTTP server.

![](../assets/demo.png)

## Trace Viewer

The trace viewer reads trace files produced by the lightweight trainer at:

```txt
generated/<version>/<qid>.trace.json
```

With the current local lightweight config, the default generated directory is typically:

```txt
/tmp/areal/experiments/logs/root/<experiment_name>/<trial_name>/generated
```

Example for the default Qwen3 local run:

```bash
bash launch_trace_viewer.sh \
  /tmp/areal/experiments/logs/root/asearcher-light-qwen3/run1/generated \
  127.0.0.1 \
  8765
```

Or directly:

```bash
python3 light_trace_server.py \
  --trace-dir /tmp/areal/experiments/logs/root/asearcher-light-qwen3/run1/generated \
  --host 127.0.0.1 \
  --port 8765
```

Then open `http://127.0.0.1:8765`.
