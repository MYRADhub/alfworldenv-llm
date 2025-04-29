# ALFWorld Exploration Agent

An intelligent exploration agent for ALFWorld that uses Large Language Models (LLMs) for decision-making.  
This project enhances the standard ALFWorld agent with:
- **Chain-of-Thought Reasoning**  
- **Memory-Based Exploration**  
- **Confidence-Based Stopping**  

The goal is to efficiently explore complex virtual homes simulated in AI2-THOR, leveraging reasoning and memory to improve object-finding tasks.

---

## Features

- **Chain-of-Thought Reasoning:**  
  The LLM thinks step-by-step about what it knows so far before choosing its next action.
  
- **Memory-Based Exploration:**  
  The agent remembers previously explored objects and locations to avoid redundant actions.
  
- **Confidence-Based Stopping:**  
  The LLM estimates confidence in its guesses and can stop exploration early when it is sufficiently sure, saving time and resources.

---

## Installation

Follow these steps to install and set up the environment:

### Requirements
- Python 3.9
- `git`
- `virtualenv` (or any virtual environment tool)
- An OpenAI API Key

### Step-by-Step Setup

1. **Clone the ALFWorld repository:**

   ```bash
   git clone https://github.com/alfworld/alfworld.git alfworld
   cd alfworld
   ```

2. **Create and activate a virtual environment:**

   Using `virtualenv`:

   ```bash
   virtualenv -p $(which python3.9) --system-site-packages alfworld_env
   source alfworld_env/bin/activate
   ```

   Alternatively, you can use `python3.9 -m venv alfworld_env`.

3. **Install ALFWorld and its dependencies:**

   ```bash
   pip install -e .[full]
   pip install alfworld[vis]
   ```

4. **Download ALFWorld data:**

   ```bash
   export ALFWORLD_DATA=<your_storage_path>
   python scripts/alfworld-download
   ```

   Replace `<your_storage_path>` with your preferred local directory for storing the ALFWorld datasets.

5. **Configure the OpenAI API Key:**

   Navigate back to the project root:

   ```bash
   cd ..
   ```

   Create a `.env` file:

   ```bash
   touch .env
   ```

   Edit `.env` and insert your API key:

   ```
   OPENAI_API_KEY=<your_openai_api_key>
   ```

   Replace `<your_openai_api_key>` with your actual OpenAI key.

---

## Usage

Once installation is complete:

1. Activate your environment (if not already activated):

   ```bash
   source alfworld_env/bin/activate
   ```

2. Run the evaluation script:

   ```bash
   bash run_eval.sh
   ```

The agent will start exploring and interacting with the environment, using the LLM for reasoning and decision-making at each step.

---

## Project Structure

- `alfworld/` — Main environment code
- `agents/` - Folder with all agents code
- `run_eval.sh` — Script for running evaluation
- `results/` — Folder where evaluation output is saved
- `logs/` - Folder of logs from run_eval.sh
- `eval_attributes/` - Folder with different extra_attributes.json files used in evaluation
- `utils/` - Some helper files that I used as references or to get all objects in the environment
- `.env` — Your API keys and private environment variables (do not share this file)

---

## Troubleshooting

- Make sure **Python 3.9** is being used.  
- Ensure you have a working **OpenAI API key** with sufficient quota.
- If `pip install` fails, try upgrading your `pip`, `setuptools`, and `wheel`:
- If ALFWorld visualizations do not work, double-check that `alfworld[vis]` is installed correctly.