# 两足人形 RL 平台

MuJoCo 里的一具人型两足。交互窗口里用 WASD 和鼠标看它，左键发炮弹。强化学习接口已经留好。

Python 3.14 没有 PyBullet 轮子。MuJoCo 的接触也更适合两足站立和爬起，所以用 MuJoCo。

## 运行

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m robot_rl
```

| 操作 | 作用 |
| --- | --- |
| WASD | 水平移动摄像机 |
| 鼠标 | 转向 |
| 左键 | 沿准心发射炮弹 |
| 空格 / C | 上升 / 下降 |
| Shift | 加速 |
| Tab | 释放或重新捕获鼠标 |
| R | 重置到站立 |
| Esc | 退出 |

不加载策略时动作为 0：底层 PD 保持站立关节角。这不是平衡策略，炮弹可以把他打倒。打倒之后关节仍会往站立角收，爬起要靠后面的策略。

## 训练接口

```python
from robot_rl import HumanoidBalanceEnv

env = HumanoidBalanceEnv()
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

env.reset(options={"pose": "supine"})  # 或 prone，用来做爬起
env.fire_projectile(origin, direction)
```

- 动作 `[-1, 1]`，0 是站立姿态，±1 顶到该关节限位。顺序见 `robot_rl/robot.py` 的 `JOINTS`。
- 观测布局：`robot_rl.robot.describe_observation()`。`pressure` 是脚、小腿、大腿、手、前臂、上臂、骨盆、胸口和头的法向接触力，除以体重。
- 炮弹打到身上的这一步就算力，然后立刻消失，不再压在身上。
- 奖励在 `robot_rl/rewards.py`。站立课按 Isaac Lab 的速度任务，命令固定为 0。水平速度和转向各用 exp 核对着 0，权重 1，再乘以躯干立起来的程度。竖向速度 −2，横滚俯仰角速度 −0.05，躯干倾斜 −1，动作变化 −0.005，关节加速度和力矩是 1e-7 这一档，脚滑 −0.1，脚以外的部位着地每个 −1，关节偏离站立角 −0.1。爬起课先只做发现：高度、高度增量、立起、双脚承重和脚力增量。不扣平滑、力矩、速度和着地。`getup-plus` 把这两套加在一起。开局一成站着、四成趴着、四成仰躺、一成任意姿态（一半贴地，一半从空中落下）。不加向上的拉力。骨盆或胸口碰到地面且力超过 1 N，站立课这一局结束；getup-plus、getup-human 和 impact 不因此结束。小球打在胸口上不算落地。`info["reward_terms"]` 里有各项。`getup-human` 和 `impact` 使用 DeepMimic 相位跟踪，见下方说明。
- 训练按课程选，一次只跑一个。默认 256 个人形：十几个进程，每个进程里像 Unity 那样串行推十几个。查看器仍然只有一个。策略更新在 GPU（没有 CUDA 就用 CPU）。模型写到 `runs/humanoid_ppo_<课程>`。中途检查点在 `runs/checkpoints`，每个课程只留最近 5 个；成功率最高的那一版另外写成 `humanoid_ppo_<课程>_best.zip`，不会被清掉。KL 上限由 `robot_rl/train.py` 的 `TARGET_KL` 配置，超过其 1.5 倍就停下本轮剩下的更新，batch 2048。站立课最近一万局全成功才提前停。getup-plus 一万局里高于 97% 就停。getup-human 和 impact 跑满预算，另做固定协议评估保存最佳权重。

```powershell
.\.venv\Scripts\python.exe -m robot_rl.train --curriculum stand
.\.venv\Scripts\python.exe -m robot_rl.train --curriculum getup-plus
.\.venv\Scripts\python.exe -m robot_rl.train --curriculum getup-human
.\.venv\Scripts\python.exe -m robot_rl.train --curriculum impact
```

`stand` 不发炮弹，并且不加熵奖励。`getup-plus` 开局一成站着、四成趴着、四成仰躺、一成任意姿态，发现奖励上再叠站立奖励，倒地 8 秒才失败。`impact` 每局从站立开场，立刻水平打出且只打出一发球；动量从基准的 0.02–0.1 倍逐步增加到 1–5 倍，速度保持 40 m/s，按本次新增训练步数计算进度。扛住球也可成功。成功不提前重置；到 30 秒才按最后的稳定保持状态判成功或失败，然后重置。除了 `getup-plus`、`getup-human` 和 `impact`，骨盆或胸口着地这一局立刻失败。
- 曲线用 SwanLab 本地看板，训练开始后打开 http://127.0.0.1:5092 。上报名是中文：`训练/价值损失`、`采样/成功率`、`奖励/` 下是各项奖励，`动作/` 是参考匹配，这两类凑满两局再记一个点。`训练/相对初始KL` 是当前策略相对开训（或 load 后、尚未更新）冻结副本的 KL。`评估/` 是固定评估，`恢复/保持比例` 是严格站稳的比例。日志在 `runs/swanlog`，不上传。看板没起来时再跑 `.\.venv\Scripts\python.exe -m swanlab watch runs/swanlog`。
- 不选课程时，跌倒不结束回合。只有数值发散才 `terminated`，超时是 `truncated`。选了课程以后，倒地超时也是 `truncated`，`info["is_success"]` 标这一局有没有站到时间上限。getup-human 在 20 秒、impact 在 30 秒统一检查结局，均为 `terminated`；数值发散提前失败，复位清空所有保持状态。
- 查看器可以接策略：`run(policy)`，策略实现 `act(observation) -> action`。

物理步长 2 ms，控制频率 50 Hz。

### getup-human：只保留 DeepMimic

此课程借鉴 [DeepMimic（SIGGRAPH 2018 / ACM TOG）](https://xbpeng.github.io/projects/DeepMimic/index.html) 的模仿奖励和参考状态初始化；策略只接收身体状态，不采用论文的相位和技能条件输入。移除 HumanUP 的跟踪权重 8、十项正则及随进度增加的正则课程。旧 `stand`、`getup-plus` 保留供旧模型使用。

- 输入统一整理为 189 维：骨盆高度 1、骨盆朝向 4、骨盆线速度和角速度各 3、其余 11 个身体部位各 13 维（位置 3 / 朝向 4 / 线速度 3 / 角速度 3）、上一动作 20、接触压力 15，即 `11 + 11×13 + 20 + 15 = 189`。骨盆相对自身的位置恒为零，删除；关节角与速度、基座速度与重力不再和全身运动状态重复输入。上一动作和接触信息保留，它们不能从身体姿态直接还原。删除旧 `home` 回原点信息。策略和价值网络均不接收技能、相位、任务编号、参考姿态或期望动作；参考时钟只用于训练奖励。
- 输出仍是 20 个铰链关节的 PD 目标角（归一化到现有动作范围），与论文的关节目标控制语义一致。论文人体的 36 维动作对应另一种骨骼结构，不能直接套在本模型上。
- 策略和价值网络使用 1024 / 512 ReLU 隐藏层；高斯探索标准差固定，不随 PPO 更新学习。控制频率保留本平台的 50 Hz，论文使用 30 Hz。
- 训练开局从两条参考动作均匀抽取时间，并初始化对应的关节角、根姿态和速度（RSI）。显式 `reset(options={"pose": "prone"})` 或 `supine` 则从该参考的第一帧开始，固定评估关闭 RSI。
- 只对所选动作当前时刻计分。姿态、速度、手脚和质心项权重分别为 0.65 / 0.10 / 0.15 / 0.10，使用论文的指数核及平方误差求和；铰链角误差替代球关节的四元数误差，姿态项也包含根朝向误差。手脚和质心去掉根的水平平移并对齐朝向，保留世界高度。
- 参考结束后保持站姿，不循环播放起身。起身中允许躯干接地；getup-human 参考结束后若躯干再次接地则提前失败。impact 保留受击后的起身窗口，并在倒地时选择一次恢复动作，不逐帧重新选参考。

参考沿用官方公开起身数据，采用原始播放时长，并加一秒向本模型站姿的过渡，趴地约 4.03 秒、仰躺约 4.77 秒。重定向仍是近似运动学转换，尚未做动力学可行性优化。这是只有身体输入的参考模仿实验，不是原版 DeepMimic：去掉相位后，同一身体状态在不同参考时刻可能对应不同奖励目标，存在训练冲突，需要用实际恢复与回放验证。来源和构建方法见 [动作说明](robot_rl/assets/motions/README.md)。

旧 89 / 157 / 160 / 245 维模型都不能载入这版课程。请从头训练：

```powershell
.\.venv\Scripts\python.exe -m robot_rl.train --curriculum getup-human --timesteps 200000000
```

新版模型使用独立的 `_deepmimic_state` 文件名，不覆盖旧模型。起身训练完成后，可载入新版最佳模型训练 impact：

```powershell
.\.venv\Scripts\python.exe -m robot_rl.train --curriculum impact --timesteps 300000000 --load runs\checkpoints\humanoid_ppo_getup-human_deepmimic_state_best.zip
```

两课共用项目自己的严格恢复评估：骨盆高度 ≥0.8 m、骨盆和胸口 upright ≥0.9、双脚当前接触、脚之外无 >1 N 支撑、基座速度 <0.25 m/s、角速度 <0.6 rad/s。连续满足 2 秒并再保持 3 秒才算恢复；失稳重新计数。getup-human 整局 20 秒，impact 30 秒，局末仍满足保持才成功。评估标准不额外进入模仿奖励。impact 每局仅开场一球，扛住也能成功；需真实击中。

开始训练及每新增约 256 万步做一次 8 个固定种子的完整回合评估。起身涵盖站立、趴地、仰躺和任意姿态；impact 使用基准动量。按成功率优先、再按参考匹配与动作变化选最佳模型。查看 `eval/recovery_success_rate`、`eval/motion_imitation`、`eval/action_delta_rms`，并用新种子和回放验证动作。代码检查和短训练通过不代表已经学会自然起身。

查看新版策略：

```powershell
.\.venv\Scripts\python.exe -m robot_rl.play --policy runs\checkpoints\humanoid_ppo_getup-human_deepmimic_state_best.zip
```

查看器提供与训练一致的 189 维观测，直接调用策略。它不加载参考动作、不选择技能，也不维护动作时钟。旧 89 / 157 / 245 维策略继续使用各自输入；此前含引导输入的 160 维模型明确拒绝加载。

观测按身体状态、接触和动作历史整理，不按维度数量堆叠。全身位置和朝向仍受骨骼运动学约束，不应宣称 189 个标量彼此独立或是数学上的最小状态。此布局的效果仍需通过训练和固定评估验证。
