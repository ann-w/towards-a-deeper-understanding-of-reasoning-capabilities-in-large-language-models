# Towards-a-deeper-understanding-of-reasoning-capabilities-in-large-language-models

This is the code repository accompanying the paper 'Towards a deeper understanding of reasoning capabilities in large language models'.

# 1. Installation

You can install the project with Docker or conda.

## Option 1: Conda

Make sure to have conda installed. The installation consists of two steps. First install the SmartPlay package, then install the requirements.txt in the root of this project.

### **Install Smartplay**

1. Follow the steps in `libs/smartplay/README.md` for the installation steps.

2. Verify your installation by:

    ```bash
    pip show smartplay
    ```

    > **Note:**
    > The gym version in Smartplay relies on a specific pip version, you can run:
    > `pip install --upgrade pip==23.*`

3. From the root folder, activatate the virtual env:

    ```bash
    conda activate SmartPlay
    ```

4. Install requirements for LLM agent

    ```bash
    pip install -r requirements.txt
    ```

5. Set the PYTHONPATH variable, from the root directory run:

    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)/src
    ```

## Option 2: Docker

Make sure to have Docker installed and start Docker.

1. Build the docker container:

    ```bash
    docker buildx build --platform=linux/amd64 -t llm-agent:latest .
    ```

2. Run the container:

    ```bash
    docker run -it --rm -p 0:80 llm-agent:latest
    ```

# 2. Setup

Save .env.template as .env and fill in your openAI API Key and/or Huggingface Key. Then load it in your environment by using the following command:

```bash
source .env
```

This repository uses wandb for logging. You need to login first with you API key:

```bash
wandb login
```

# 3. Usage


1. Set the desired parameters in the `src/config/experiment_settings/experiment_settings.yaml`
2. Run the experiments from root directory:

```bash
./run_experiment.sh
```

### Run ollama

if you want to run the models using ollama, first pull the model onto your local machine before you run the experiments.

```
ollama serve
```

and then

```bash
ollama run <model-name>
```

### Run huggingface

You can run the huggingface models either locally or using their inference api. You need to have an account and login using:

```bash
huggingface-cli login
```
