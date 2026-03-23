import sys
import os
import modal

app = modal.App("tayavision-eval-checkpoint")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file("pyproject.toml", remote_path="/root/project/pyproject.toml", copy=True)
    # Install everything into the system path
    .run_commands(
        "pip install --upgrade pip",
        "cd /root/project && pip install . vllm ray 'transformers>=4.46.0' lm-eval"
    )
    # Copy full project so that imports all resolve inside the container.
    .add_local_dir("evaluation", remote_path="/root/project/evaluation", copy=True)
    .add_local_dir("models", remote_path="/root/project/models", copy=True)
    .add_local_dir("config", remote_path="/root/project/config", copy=True)
    .add_local_dir("src", remote_path="/root/project/src", copy=True)
    .add_local_dir("pipeline", remote_path="/root/project/pipeline", copy=True)
    .add_local_dir("scripts", remote_path="/root/project/scripts", copy=True)
)

# Persistent volume for results in the cloud
results_volume = modal.Volume.from_name("tayavision-results", create_if_missing=True)

@app.function(
    image=image,
    gpu="H100",
    volumes={"/results": results_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600 * 4,
)
def run_evaluation(task: str, batch_size: str = "auto", log_samples: bool = False, apply_chat_template: bool = False, limit: int = None):
    # Setup project environment inside the container
    sys.path.insert(0, "/root/project")
    os.chdir("/root/project")

    # Prepare the checkpoint inside the container (download + assemble), saves a dir to be loaded
    prepared_model_dir = "/tmp/eval_model"
    sys.argv = [
        "prepare_checkpoint.py",
        "--output-dir", prepared_model_dir,
        "--device", "cuda",
    ]
    from scripts.prepare_checkpoint import main as prepare_main
    prepare_main()

    # Define a unique output directory on the volume to avoid mount conflicts
    # and to organize results by task as requested
    cloud_output_dir = f"/results/modal_{task}"
    os.makedirs(cloud_output_dir, exist_ok=True)

    # We call your existing evaluation script as if it were running natively
    from evaluation.run_eval import main

    # Pass arguments to the argparse in run_eval.py
    sys.argv = [
        "run_eval.py",
        "--task", task,
        "--model-name", prepared_model_dir,
        "--backend", "hf-multimodal",  # vllm doesn't support custom VLM architectures; hf-multimodal uses HF generate() directly
        "--batch-size", batch_size,
        "--output-dir", cloud_output_dir
    ]

    if log_samples:
        sys.argv.append("--log-samples")

    if apply_chat_template:
        sys.argv.append("--apply-chat-template")

    if limit:
        sys.argv.extend(["--limit", str(limit)])

    main()

    # Read generated results to send them back locally (recursive)
    results_data = {}
    for root, dirs, files in os.walk(cloud_output_dir):
        for filename in files:
            if filename.endswith(".json") or filename.endswith(".jsonl"):
                abs_path = os.path.join(root, filename)
                # Keep the 'modal_{task}' part in the relative path for local organization
                rel_path = os.path.relpath(abs_path, "/results")
                with open(abs_path, "r") as f:
                    results_data[rel_path] = f.read()

    results_volume.commit()
    return results_data


# `modal run scripts/modal_eval_checkpoint_en.py`
@app.local_entrypoint()
def main(task: str, batch_size: str = "auto", log_samples: bool = False, apply_chat_template: bool = False, limit: int = None):

    # We trigger the remote call that runs in the cloud with .remote()
    print("Initializing cloud GPU...")
    results_dict = run_evaluation.remote(
        task=task,
        batch_size=batch_size,
        log_samples=log_samples,
        apply_chat_template=apply_chat_template,
        limit=limit
    )

    # Write the results back to the local results directory
    local_base_dir = "evaluation/results"
    os.makedirs(local_base_dir, exist_ok=True)

    for rel_path, content in results_dict.items():
        local_path = os.path.join(local_base_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as f:
            f.write(content)
        print(f"Synced result: {local_path}")

    print("\nEvaluation complete. Results are stored in evaluation/results/")
