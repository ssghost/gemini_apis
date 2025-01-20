from typing import Dict, Union
from huggingface_hub import get_safetensors_metadata
import argparse
import sys

bytes_per_dtype: Dict[str, float] = {
    "int4": 0.5,
    "int8": 1,
    "float8": 1,
    "float16": 2,
    "float32": 4,
}


def calculate_gpu_memory(parameters: float, bytes: float) -> float:
    memory = round((parameters * 4) / (32 / (bytes * 8)) * 1.18, 2)
    return memory


def get_model_size(model_id: str, dtype: str = "float16") -> Union[float, None]:
    try:
        if dtype not in bytes_per_dtype:
            raise ValueError(
                f"Unsupported dtype: {dtype}. Supported types: {list(bytes_per_dtype.keys())}"
            )

        metadata = get_safetensors_metadata(model_id)
        if not metadata or not metadata.parameter_count:
            raise ValueError(f"Could not fetch metadata for model: {model_id}")

        model_parameters = list(metadata.parameter_count.values())[0]
        model_parameters = int(model_parameters) / 1_000_000_000  # Convert to billions
        return calculate_gpu_memory(model_parameters, bytes_per_dtype[dtype])

    except Exception as e:
        print(f"Error estimating model size: {str(e)}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Estimate GPU memory requirements for Hugging Face models"
    )
    parser.add_argument(
        "model_id", help="Hugging Face model ID (e.g., Qwen/Qwen2.5-7B-Instruct)"
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=bytes_per_dtype.keys(),
        help="Data type for model loading",
    )

    args = parser.parse_args()
    size = get_model_size(args.model_id, args.dtype)

    print(
        f"Estimated GPU memory requirement for {args.model_id}: {size:.2f} GB ({args.dtype})"
    )


if __name__ == "__main__":
    main()
