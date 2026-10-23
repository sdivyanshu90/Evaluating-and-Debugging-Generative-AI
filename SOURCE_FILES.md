**Notebook source files**

The source companions below preserve notebook cell order and explanations in readable code blocks. Python uses `# %%` cell markers; TypeScript uses `// %%`. The original notebooks retain their saved visualizations and course context. GitHub language statistics exclude `.ipynb` files through `.gitattributes` and count eligible implementation source normally.

Use the original notebook's environment, dependency versions, datasets, credentials and working directory. This conversion does not supply missing course helpers or fill exercise placeholders. It does not execute training, external APIs, shell commands or setup cells during validation. Review setup and execution cells before running a whole lesson.

Run ordinary Python companions with `python "path/to/lesson.py"`, or open their cells in an editor supporting Python percent scripts. Files marked **IPython** retain notebook magics and shell commands: install IPython in the lesson environment and run `ipython "path/to/lesson.py"`. Files marked **async** have an `async def main()` entry point; run the whole script to preserve its event loop. Script validation checks Python syntax, not availability or behavior of the original dependencies.

TypeScript companions preserve the Deno runtime identified by their notebooks. Run them in the original Deno environment with the permissions and dependencies required by that lesson. Repeated notebook declarations use mutable bindings and reassignment in the script. React component variants are separate `.tsx` modules in the adjacent `.components` directory; the lesson's frontend-injection helpers still require the original course environment. No Python translation is implied for TypeScript code.

| Notebook | Source companion | Execution notes |
|---|---|---|
| [01_intro_starter.ipynb](01_intro_starter.ipynb) | [01_intro_starter.py](01_intro_starter.py) | Python |
| [02_diffusion_training_starter.ipynb](02_diffusion_training_starter.ipynb) | [02_diffusion_training_starter.py](02_diffusion_training_starter.py) | Python |
| [03_diffusion_sampling_starter.ipynb](03_diffusion_sampling_starter.ipynb) | [03_diffusion_sampling_starter.py](03_diffusion_sampling_starter.py) | Python |
| [04_llm_eval_starter.ipynb](04_llm_eval_starter.ipynb) | [04_llm_eval_starter.py](04_llm_eval_starter.py) | Python |
| [05_train_llm_starter.ipynb](05_train_llm_starter.ipynb) | [05_train_llm_starter.py](05_train_llm_starter.py) | Python |
