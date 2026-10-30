# 两足人形 RL 平台

MuJoCo 两足机器人，20 个 PD 关节目标。交互窗口使用 WASD 和鼠标，左键发射球，R 复位，Tab 释放鼠标，Esc 退出。

## 运行

```powershell
.\.venv\Scripts\python.exe -m robot_rl
```

stand 与历史模型物理步长 2 ms，新版 getup/impact 为 1 ms；控制频率均为 50 Hz。未加载策略时零动作保持默认站立关节目标。

## 课程

只保留三个课程：

- `stand`：不发球，训练站立平衡，保留原 89 维输入和站立奖励。
- `getup`：HumanUP 思路的两阶段起身，默认第一阶段自主发现；没有其他起身课程别名。
- `impact`：站立开场一发球，训练受击恢复；扛住球也算成功。

旧课程名称已删除；历史模型文件没有被删除或改写。

## getup：HumanUP

依据 [HumanUP（RSS 2025）](https://humanoid-getup.github.io/) 的先发现、后精修方法，[论文](https://arxiv.org/html/2502.12152v2) 与 [官方实现](https://github.com/RunpeiDong/HumanUP)。不再使用外部人体动作素材、DeepMimic 模仿奖励或参考姿势随机初始化。

### 通用输入

保留已有的 189 维状态作为完整前缀：骨盆高度、朝向、线角速度，11 个非骨盆部位的位置/朝向/线角速度，20 维上一动作和 15 维接触压力。

追加过去三步的紧凑观测，每步 65 维：角速度 3、roll/pitch 2、20 个关节角、20 个关节速度、20 维上一动作。因此总输入是 `189 + 3×65 = 384` 维。历史按时间从旧到新排列，包含过去三步，不重复拼入当前帧；开局复制初始观测填充历史，复位清空上一局数据。原 189 维内容没有删减。新版固定缩放角速度 ×0.25、历史关节速度 ×0.05，并将观测裁剪到 ±100；训练和查看器使用相同规则。

这些数据均来自机器人自身状态和动作历史。没有技能编号、相位、参考动作、目标关节角、任务阶段或真实环境参数输入。查看器使用相同的历史更新规则，推理不加载参考文件。未来在同一机器人上换任务时可沿用输入布局，按任务重新训练奖励。

输出仍是 20 个归一化 PD 关节目标。模型使用项目适配的 256/256 ELU MLP 与 SB3 PPO，动作标准差正常学习。有界高斯动作使用 tanh，采样、概率密度与推理使用相同变换；高斯潜在均值软限幅到 ±2、log_std 限于 [-5, -0.7]，保留 20 维动作头，避免输出远超限位后被硬裁成常数。

### 第一阶段：自主发现

```powershell
.\.venv\Scripts\python.exe -m robot_rl.train --curriculum getup --timesteps 200000000
```

默认 `getup` 使用本项目的新恢复奖励和辅助课程，不再把它称作 HumanUP 复现。30% 从站姿练保持，70% 从仰躺练恢复；初期向骨盆施加至多 0.6 倍体重的向上辅助，随高度在站高的 75% 前降为零，最后五秒强制无辅助。每个环境最近 16 次倒地回合至少有 8 次且成功率达到 75% 后，辅助上限按 0.6→0.4→0.2→0 逐级下降；完全撤掉辅助后才加入趴地，最多 40%。站姿回合不推进难度。

奖励使用本机站高归一化的高度/直立误差、折扣一致进展，以及接近站立后的双脚支撑和静止奖励。未站起没有静态抬头、脚间距或受力增加的正奖励；不起身阶段没有对称和站姿关节约束，速度惩罚只在接近站立时增强。无辅助连续稳定五秒并保持至局末才算成功；显式测试姿势始终无辅助。输入、网络和课程名不变。
模型保存为 `runs/humanoid_ppo_getup.zip`；最佳模型单独保存为 `runs/checkpoints/humanoid_ppo_getup_best.zip`。旧 839 维输入模型不能直接续训新版；保存的契约标识会校验缩放和物理设置。旧模型仍可用查看器按旧规则回放。

### 导出本机器人成功轨迹

```powershell
.\.venv\Scripts\python.exe -m robot_rl.export_humanup --model runs\checkpoints\humanoid_ppo_getup_best.zip --output runs\humanup_reference.npz
```

导出会实际执行趴地和仰躺恢复，要求两种姿势均连续满足严格稳定条件五秒后才写文件。未找到成功动作时拒绝导出，也不会覆盖已有参考。导出记录关节角、头部高度、时间、关节顺序与源模型校验值；默认将发现的动作放慢八倍。不把长期静态站立尾段一起拉长。

### 第二阶段：精修

```powershell
.\.venv\Scripts\python.exe -m robot_rl.train --curriculum getup --stage refinement --reference runs\humanup_reference.npz --load runs\checkpoints\humanoid_ppo_getup_best.zip --timesteps 300000000
```

第二阶段使用导出的本机器人轨迹，以关节跟踪及官方的速度、力矩、动作变化与关节限位项精修。初始关节姿势增加扰动。无参考文件或文件格式不兼容时直接报错，不会回退到旧素材。

**第二阶段仍按参考时间推进奖励，这是官方 HumanUP 也采用的机制。相位和参考目标没有进入策略，三步历史也不能保证完全推断隐藏参考时刻。** 如果机器人显著偏离轨迹，时间对齐冲突仍可能存在。默认第一阶段不使用这个时钟，不能把换成 HumanUP 说成彻底解决一切轨迹对齐问题。

参考较长时自动延长第二阶段局长，至少留出完整动作时长加五秒稳定保持。精修和发现共享 384 维输入，权重可继续训练；`--timesteps` 表示累计总步数。两个阶段均使用 `getup` 文件名；同时保留最终模型时用 `--save` 显式指定路径。

## 抗击打与评估

```powershell
.\.venv\Scripts\python.exe -m robot_rl.train --curriculum impact --load runs\checkpoints\humanoid_ppo_getup_best.zip --timesteps 300000000
```

impact 默认使用发现阶段任务奖励。也可传 `--stage refinement --reference ...` 使用已验证轨迹精修。每局只有开场一发球；动量从弱到强，按本次新增训练步数计算进度。

两课沿用严格稳定评估：骨盆高度 ≥0.8 m、骨盆与胸口直立程度 ≥0.9、双脚当前接触、非脚部无 >1 N 支撑、线速度 <0.25 m/s、角速度 <0.6 rad/s。连续满足两秒进入保持，再保持三秒完成，任何失稳清零。getup 默认整局 20 秒，impact 30 秒，局末必须仍满足连续稳定条件。完成后不立即复位；没有单独的局末成功/失败加分。数值发散提前失败。

固定评估在开训和每新增约 256 万步进行，起身课优先比较倒地开局的严格成功率，再比较整体成功率、姿态质量与动作抖动；站立、仰躺、趴地、随机姿势分别记录成功率，不用总奖励代替恢复能力。训练曲线在 SwanLab 本地看板 http://127.0.0.1:5092 ，日志写入 `runs/swanlog`。

查看策略：

```powershell
.\.venv\Scripts\python.exe -m robot_rl.play --policy runs\checkpoints\humanoid_ppo_getup_best.zip
```

启动训练仅记录 SwanLab 日志，不自动启动看板服务或打开浏览器。需要时手动运行：

```powershell
.\.venv\Scripts\python.exe -m swanlab watch runs\swanlog --host 127.0.0.1 --port 5092
```

然后手动访问 http://127.0.0.1:5092 。查看器从原始 SB3 zip 读取 actor 权重，用原生 PyTorch MLP 推理；跳过优化器、价值网络和 rollout buffer，不生成其他格式的模型缓存。续训前只读元数据检查兼容性，完整训练权重和优化器状态仅在正式加载时读取一次。

## 实现边界

这是本项目 20 关节 MuJoCo 模型上的 HumanUP 适配。论文使用 G1、Isaac Gym、分开的翻身/站起策略、不同碰撞网格和 ROA 环境隐变量；这里用一个起身策略覆盖趴地和仰躺，保留当前碰撞模型与硬件限位，物理频率已对齐 1 kHz。未伪造 ROA 隐变量，也没有宣称完成地形随机化或实机迁移。

新发现阶段参考 [HoST](https://taohuang13.github.io/humanoid-standingup.github.io/) 的初期垂直辅助与逐步撤除思路；奖励公式、权重和成功驱动课程是本项目设计，并非 HoST 或 HumanUP 的复现。保留单策略、384 维三帧物理输入、256/256 ELU 网络；相位、辅助强度、课程级别和参考动作不作为输入。

旧 HumanUP 精修模式仍可显式使用 `--stage refinement --reference ...`，但默认起身奖励已独立替换；不强制导出轨迹或进入第二阶段。旧日志、检查点保留。模型与实验名仍为 getup，不加版本后缀。建议新奖励从头训练，不能凭静态奖励检查或短训练测试断言已经学会起身。

MuJoCo 当前在 CPU 运行，GPU 负责策略计算与 PPO 更新。看板记录采样秒数、采样步速、更新秒数和采样占比；GPU 利用率不等于整体训练吞吐。训练不会自动打开看板或浏览器。

固定局长结束使用 Gymnasium `truncated=True`，保留成功/失败评估标记，并让 SB3 从 `terminal_observation` 补偿后续价值；仿真发散或非有限状态才是 `terminated=True`。这样与 HumanUP 官方 `time_outs` 的价值补偿语义对齐，避免所有环境同时到达局长时把价值目标突然截为零。该修正不改变奖励公式、网络输入或成功标准。
