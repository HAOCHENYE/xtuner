.. xtuner documentation master file, created by
   sphinx-quickstart on Tue Jan  9 16:33:06 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |checked| unicode:: U+2713
.. |unchecked| unicode:: U+2717

Welcome to XTuner V1 English Documentation
==========================================

.. figure:: ./_static/image/logo.png
  :align: center
  :alt: xtuner
  :class: no-scaled-link

.. raw:: html

   <p style="text-align:center">
   <strong>LLM One-Stop Toolbox
   </strong>
   </p>

   <p style="text-align:center">
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/InternLM/xtuner" data-show-count="true" data-size="large" aria-label="Star">Star</a>
   <a class="github-button" href="https://github.com/InternLM/xtuner/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
   <a class="github-button" href="https://github.com/InternLM/xtuner/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
   </p>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting Started

   get_started/index.rst

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Pretraining & Fine-tuning

   pretrain_sft/tutorial/index.rst

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Reinforcement Learning

   rl/tutorial/rl_grpo_trainer.md

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Advanced Tutorial

   pretrain_sft/advanced_tutorial/index.rst
   rl/advanced_tutorial/index.rst

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Benchmark

   benchmark/index.rst

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Legacy Documentation

   legacy_index.rst

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API

   Pretrain & SFT Trainer  <api/trainer>
   Config <api/config>
   RL Trainer <api/rl_trainer>
   RL Config <api/rl_config>
   Loss Context <api/loss_ctx>


XTuner V1 is a next-generation training framework purpose-built for trillion-scale Mixture-of-Experts (MoE) models. Compared with conventional 3-D parallel training stacks, XTuner V1 is deeply re-engineered for the mainstream MoE workloads that dominate today’s research landscape.

🚀 Speed Benchmark
==================================

.. figure:: ../assets/images/benchmark/benchmark.png
   :align: center
   :width: 90%

Core Features
=============
**📊 Dropless Training**

- **Effortless scaling without tedious tuning:** No expert parallelism needed for MoE models up to 200B; a 600B MoE training only requires intra-node expert parallelism.
- **Optimized parallel plan:** Smaller expert-parallel dimensions than classic 3-D setups, enabling more efficient dropless training.

**📝 Long Sequence Support**

- **Memory Efficient Design:** An advanced combination of memory optimizations enables a 200B MoE model training with 64K sequence length without using sequence parallelism.
- **Flexible Extension Capability:** Full support for DeepSpeed Ulysses sequence parallelism for linear length scaling.
- **Stable and Reliable:** robust to expert-load imbalance under long-context training, maintaining stable performance.

**⚡ Superior Efficiency**

- **Trillion-parameter ready:** Stable training for MoE models up to 1 T parameters.
- **Breaking the throughput wall:** The first framework to outperform traditional 3-D parallelism solution with FSDP on MoE models beyond 200B.
- **Hardware-tuned:** Training efficiency surpasses NVIDIA H800 on Ascend A3 NPU supernodes.


.. figure:: ../assets/images/benchmark/structure.png
   :align: center
   :width: 90%
   :alt: Performance comparison


🔥 Roadmap
==========

XTuner V1 keeps pushing the frontier on pre-training, supervised fine-tuning and reinforcement learning for trillion-scale MoE models, with special focus on Ascend NPU co-design.

🚀 Training Engine
-----------

Our vision is to make XTuner V1 a universal backend that plugs seamlessly into a broader open-source ecosystem.

+------------+-----------+----------+-----------+
|   Model    |  GPU(FP8) | GPU(BF16)| NPU(BF16) |
+============+===========+==========+===========+
| Intern S1  |    ✅     |    ✅    |    ✅     |
+------------+-----------+----------+-----------+
| Intern VL  |    ✅     |    ✅    |    ✅     |
+------------+-----------+----------+-----------+
| Qwen3 Dense|    ✅     |    ✅    |    ✅     |
+------------+-----------+----------+-----------+
| Qwen3 MoE  |    ✅     |    ✅    |    ✅     |
+------------+-----------+----------+-----------+
| GPT OSS    |    ✅     |    ✅    |    ❌     |
+------------+-----------+----------+-----------+
| Deepseek V3|    ✅     |    ✅    |    ❌     |
+------------+-----------+----------+-----------+
| KIMI K2    |    ✅     |    ✅    |    ❌     |
+------------+-----------+----------+-----------+


🧠 Algorithm Suite
-----------

Algorithms are kept iterating rapidly and community contributions are welcome. Let's scale your algotithm recipe to unprecedented scales with XTuner V1!

**Implemented**

- ✅ **Multimodal Pre-training** - Full support for vision-language model training.
- ✅ **Multimodal Supervised Fine-tuning** - Optimized for instruction following.
- ✅ `GRPO <https://arxiv.org/pdf/2402.03300>`_ - Group Relative Policy Optimization.

**Coming Soon**

- 🔄 `MPO <https://arxiv.org/pdf/2411.10442>`_ - Mixed Preference Optimization.
- 🔄 `DAPO <https://arxiv.org/pdf/2503.14476>`_ - Dynamic Sampling Policy Optimization.
- 🔄 **Multi-round Agent Reinforcement Learning** - Advanced agent training capabilities.


⚡ Inference Engine Integration
---------------

Seamless integration with mainstream inference frameworks

* |checked| LMDeploy
* |unchecked| vLLM
* |unchecked| SGLang



🤝 Contribution Guidelines
-----------

We are grateful to every contributor who helps improve XTuner. Please see the `Contributing Guide <.github/CONTRIBUTING.md>`_ for how to get involved.

🙏 Acknowledgments
-----------

XTuner V1 is deeply inspired and fully supported by many excellent open-source projects. We express our sincere gratitude to the following pioneering works:

**Training Frameworks:**

- [Torchtitan](https://github.com/pytorch/torchtitan) - PyTorch-native distributed training framework.
- [Deepspeed](https://github.com/deepspeedai/DeepSpeed) - Microsoft's deep learning optimization library
- [MindSpeed](https://gitee.com/ascend/MindSpeed) - Ascend high-performance training acceleration library.
- [Megatron](https://github.com/NVIDIA/Megatron-LM) - NVIDIA's large-scale Transformer training framework.


**Reinforcement Learning:**

XTuner V1's reinforcement learning capabilities based on the excellent practices and experience of the following projects:

- [veRL](https://github.com/volcengine/verl) - Volcano Engine Reinforcement Learning for LLMs.
- [SLIME](https://github.com/THUDM/slime) - THU's scalable RLHF implementation.
- [AReal](https://github.com/inclusionAI/AReaL) - Ant Reasoning Reinforcement Learning for LLMs.
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) - An Easy-to-use, Scalable and High-performance RLHF Framework based on Ray.

We heartily thank all contributors and maintainers of these projects for advancing the field of large-scale model training.


🖊️ Citation
-----------

.. code-block:: bibtex

   @misc{2023xtuner,
       title={XTuner: A Toolkit for Efficiently Fine-tuning LLM},
       author={XTuner Contributors},
       howpublished = {\url{https://github.com/InternLM/xtuner}},
       year={2023}
   }

Open Source License
==========

The project is released under the `Apache License 2.0 <LICENSE>`_. Please also respect the licenses of the models and datasets you use.
