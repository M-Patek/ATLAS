# ATLAS

**Atlas of Technologies for Learning Autonomous Systems**

具身智能数据基础设施学习图谱。不是深度教程，而是技术地图——帮助理解组件连接、数据流动、关键选型。

---

## 这是什么

ATLAS是DVAS项目配套的知识基础设施，聚焦于四个维度：

| 维度 | 核心问题 | 位置 |
|------|----------|------|
| **模型层** | VLM底座、World Model、VLA如何选择 | [docs/01-foundation/](docs/01-foundation/) |
| **标注层** | 如何设计Schema、业界使用什么标准 | [docs/02-annotation/](docs/02-annotation/) |
| **算法层** | 双目+IMU、深度估计、SLAM、手势、重建 | [docs/03-perception/](docs/03-perception/) |
| **数据层** | Ego/UMI/Sim2Real/遥操作，及配套硬件 | [docs/04-data-ecosystem/](docs/04-data-ecosystem/) |

## 快速开始

### 如果你是Agent/AI助手
→ 阅读 [llms.txt](llms.txt) 获取导航指令

### 如果你是读者
1. 从 [docs/INDEX.md](docs/INDEX.md) 了解全貌
2. 根据兴趣跳入对应层
3. 每篇10-15分钟，可独立阅读

### 如果你是贡献者
→ 阅读 [AGENTS.md](AGENTS.md) 了解文档协议

## 架构总览

数据流：采集 → 感知 → 标注 → 训练 → 部署

```
┌─────────────────────────────────────────────────────┐
│  05-integration     系统整合与部署                   │
├─────────────────────────────────────────────────────┤
│  01-foundation      模型层 (VLM/World Model/VLA)    │
├─────────────────────────────────────────────────────┤
│  02-annotation      标注层 (Schema/标准/质量)       │
├─────────────────────────────────────────────────────┤
│  03-perception      感知层 (深度/SLAM/手势/重建)    │
├─────────────────────────────────────────────────────┤
│  04-data-ecosystem  数据生态 (采集范式/硬件矩阵)    │
└─────────────────────────────────────────────────────┘
```

## 当前进展

查看 [docs/_machine/status.yaml](docs/_machine/status.yaml) 了解各主题完成状态。

重点主题（按优先级）：
- [ ] FoundationStereo深度研究
- [ ] 双目+IMU协同标定
- [ ] VLA模型对比
- [ ] 硬件选型矩阵

## 关联项目

- **DVAS** — Data-driven Video Annotation System（本学习图谱的服务对象）

## 许可

MIT
