# An install script that pip installs required packages
# Also checks if the system can use CUDA for GPU acceleration

import re
import subprocess
import sys

USE_XPU = False


def check_python_version():
    major, minor = sys.version_info[:2]
    # expect 3.10.x
    if major != 3 or minor != 10:
        print(
            "This script requires Python 3.10.x. Please install the correct version and try again."
        )
        sys.exit(1)


def check_cuda():
    """Check if CUDA is available on the system."""
    try:
        # Check for NVIDIA GPU
        result = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            return None

        # Check which CUDA version is installed
        result = subprocess.run(
            ["nvidia-smi", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output = result.stdout.decode("utf-8")
        cuda_string = [line for line in output.split("\n") if "CUDA Version" in line][0]
        cuda_version = re.search(
            r"\s*CUDA Version\s*:\s*(\d+\.\d+)", cuda_string
        ).group(1)
        major_version = int(cuda_version.split(".")[0])
        minor_version = int(cuda_version.split(".")[1])
        return (major_version, minor_version)

    except Exception:
        return


def get_cuda_index_url(cuda_version):
    """Get the appropriate PyTorch index URL for the given CUDA version."""
    if cuda_version is None:
        return None
    major, minor = cuda_version

    if major == 13:
        return "https://download.pytorch.org/whl/cu130"
    elif major == 12:
        if minor >= 8:
            return "https://download.pytorch.org/whl/cu128"
        elif minor >= 6:
            return "https://download.pytorch.org/whl/cu126"

    print(
        "CUDA version not supported for PyTorch installation. Make sure you have CUDA 12.6 or higher."
    )
    return None


def install_packages():
    """Install required packages using pip."""
    specific_version_packages = [
        "numpy==1.26.4",
        "gym==0.26.2",
        "gymnasium==0.28.1",
        "opencv-python<4.10",
        "tensorflow==2.16.2",
    ]

    torch_packages = ["torch", "torchvision", "torchaudio"]

    required_packages = [
        "torchrl",
        "tyro",
        "matplotlib",
        "torchinfo",
        "torch-pruning",
        "procgen",
        "stable_baselines3",
        "tqdm",
        "kornia",
        "higher",
        "scipy",
        "joblib",
        "cloudpickle",
        "click",
    ]

    cuda_version = check_cuda()
    cuda_available = get_cuda_index_url(cuda_version)

    base_pip_command = [sys.executable, "-m", "pip", "install", "--upgrade"]

    torch_command = base_pip_command + torch_packages
    if cuda_available:
        torch_command += ["--index-url", cuda_available]
    if USE_XPU:
        torch_command += ["--index-url", "https://download.pytorch.org/whl/xpu"]

    specific_command = base_pip_command + specific_version_packages
    required_command = base_pip_command + required_packages

    print("Installing packages...")
    subprocess.check_call(specific_command)
    subprocess.check_call(torch_command)
    subprocess.check_call(required_command)
    print("Installation complete.")


if __name__ == "__main__":
    check_python_version()
    install_packages()
