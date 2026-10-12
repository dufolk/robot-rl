# 两足人形 RL 平台

MuJoCo 里的一具人型两足。交互窗口里用 WASD 和鼠标看它，左键发炮弹。强化学习接口已经留好，奖励现在是 0。

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

不加载策略时动作为 0：底层 PD 加重力补偿保持站立关节角。这不是平衡策略，炮弹可以把他打倒。打倒之后关节仍会往站立角收，爬起要靠后面的策略。

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
- 观测布局：`robot_rl.robot.describe_observation()`。
- 奖励：改 `robot_rl/rewards.py` 的 `compute_reward`。`info` 里有骨盆高度、直立程度、是否跌倒、左右脚接触、质心、机体速度、炮弹命中力。
- 跌倒不结束回合。只有数值发散才 `terminated`，超时是 `truncated`。
- 查看器可以接策略：`run(policy)`，策略实现 `act(observation) -> action`。

物理步长 2 ms，控制频率 50 Hz。
