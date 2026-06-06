"""Forgewright — an agentic CLI harness for autonomous LLM post-training.

Point it at a model and a goal (fine-tune / abliterate / quantize / serving-opt),
give it GPUs (local or over SSH), and it plans and executes the whole job
autonomously: writing code, launching jobs, recovering from failures, evaluating,
iterating, and publishing — without babysitting.

Foundation A: a hand-rolled lean agent loop + LiteLLM brain + tool layer, calling
``model-forge`` as a library for the mature ML-ops primitives.
"""

__version__ = "0.0.1"
