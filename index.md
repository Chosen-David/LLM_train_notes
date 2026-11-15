---
layout: default
title: 大模型训练的相关学习笔记
---

# 大模型训练的相关学习笔记

从并行训练的基本概念，走到源码中的参数布局、通信与更新。

## 学习笔记

[**FSDP、EP 与 TorchTitan**](notes/fsdp-ep.html)

DDP / FSDP 基础、64 GPU mesh、LAER-MoE 动态专家布局。包含正文图解、可展开问答，以及从评论截图提取的表格与代码。

## 交互图谱

[**进入 64 GPU 前向／反向训练图谱 →**](interactive/torchtitan-64gpu.html)

按模块查看 QKV、Attention、Router、专家、共享专家、梯度归约和参数生命周期，点击 GPU 查看通信伙伴。

[查看 GitHub 仓库](https://github.com/Chosen-David/LLM_train_notes)
