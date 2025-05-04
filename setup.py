from setuptools import find_packages, setup

setup(
    name="towards-a-deeper-understanding-of-reasoning-capabilities-in-large-language-models",
    version="1.0.0",
    description="A repository for the paper 'Towards a Deeper Understanding of Reasoning Capabilities in Large Language Models'",
    url="https://github.com/ann-w/self-learning-llm-agents.git",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "transformers==4.41.2",
        "bitsandbytes==0.42.0",
        "accelerate==0.31.0",
        "gym==0.21.0",
        "numpy==2.1.3",
        "tqdm==4.67.0",
        "wandb==0.18.7",
        "openai==1.33.0",
        "python-dotenv",
        "pyyaml",
        "sentencepiece==0.2.0",
        "ollama==0.3.3",
        "huggingface-hub==0.26.2",
        "matplotlib==3.9.3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
