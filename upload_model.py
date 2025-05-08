import argparse
import os
from huggingface_hub import HfApi, HfFolder # HfFolder might not be strictly needed here but good to have for token management awareness
import config # Assuming your config.py is in the same directory or accessible via PYTHONPATH

def upload_model_to_hf_hub(file_path: str, repo_id: str, hf_token: str = None):
    """
    Uploads a model file to the specified Hugging Face Hub repository.

    Args:
        file_path (str): The local path to the model file to upload.
        repo_id (str): The Hugging Face Hub repository ID (e.g., "username/repo_name").
        hf_token (str, optional): Hugging Face API token. If None, attempts to use cached token.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    file_name = os.path.basename(file_path)
    print(f"Attempting to upload '{file_name}' to Hugging Face Hub repository: {repo_id}")

    try:
        # Initialize HfApi. If hf_token is provided to upload_file, it uses that.
        # Otherwise, it relies on the cached token (from huggingface-cli login)
        # or HUGGING_FACE_HUB_TOKEN environment variable.
        api = HfApi()

        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_name, # Uploads to the root of the repo with its original name
            repo_id=repo_id,
            repo_type="model", # Ensure this matches the repo type on the Hub
            token=hf_token # Pass token if provided; if None, cached login is used
        )
        print(f"Successfully uploaded '{file_name}' to {repo_id}.")
        # Construct the URL to the uploaded file
        file_url = f"https://huggingface.co/{repo_id}/blob/main/{file_name}"
        print(f"View the file at: {file_url}")

    except Exception as e:
        print(f"Error uploading file to Hugging Face Hub: {e}")
        print("Please ensure you are logged in via 'huggingface-cli login' or have set the HUGGING_FACE_HUB_TOKEN environment variable.")
        if hf_token:
            print("If you provided a token, ensure it has 'write' permissions for the repository.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a model file to a Hugging Face Hub repository.")
    parser.add_argument(
        "model_file_path",
        type=str,
        help="The local path to the model file you want to upload."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=config.HF_REPO_ID if hasattr(config, 'HF_REPO_ID') else None,
        help="Hugging Face Hub repository ID (e.g., 'username/repo_name'). "
             "Defaults to HF_REPO_ID from config.py if available."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None, # Default to None, so it relies on cached login or env var
        help="Optional Hugging Face API token. If not provided, cached login or HUGGING_FACE_HUB_TOKEN env var will be used."
    )

    args = parser.parse_args()

    if not args.model_file_path:
        # This case should ideally be caught by argparse if model_file_path was not optional
        # For a positional argument, argparse handles it if it's missing.
        # If it were an optional arg --model_file_path, then this check would be more relevant.
        print("Error: Model file path must be provided.")
        parser.print_help()
    elif not args.repo_id:
        print("Error: Repository ID not provided and not found in config.py (HF_REPO_ID).")
        print("Please provide the --repo_id argument or ensure HF_REPO_ID is set in your config.py.")
        parser.print_help()
    else:
        upload_model_to_hf_hub(args.model_file_path, args.repo_id, args.hf_token) 