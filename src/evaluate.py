import imageio
import os

from tqdm import tqdm

from algorithm.helper import ReplayBuffer
from algorithm.tdmpc import TDMPC
from env import make_env
from pathlib import Path
from cfg import parse_cfg

from train import __CONFIG__, __LOGS__

cfg = parse_cfg(Path().cwd() / __CONFIG__)
env, agent, buffer = make_env(cfg), TDMPC(cfg), ReplayBuffer(cfg)

work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
os.makedirs(work_dir / 'frames', exist_ok=True)
agent.load(work_dir / 'agent.pth')


class VideoWriter:
    def __init__(self, path, fps=15):
        self.writer = imageio.get_writer(path, fps=fps)

    def render(self, frame):
        self.writer.append_data(frame)

    def save(self):
        self.writer.close()


class ImageWriter:
    def __init__(self, path):
        self.dir = path
        self.idx = 0

    def render(self, frame):
        imageio.imwrite(self.dir / f'{self.idx : 03d}.png', frame)
        self.idx += 1

    def save(self):
        pass


def evaluate(env, agent, step):
    """Evaluate a trained agent and optionally save a video."""

    obs, done, ep_reward, t = env.reset(), False, 0, 0

    # writer = VideoWriter(work_dir / 'video.mp4')
    writer = ImageWriter(work_dir / 'frames')
    pbar = tqdm(total=250)
    while not done:
        action = agent.plan(obs, eval_mode=True, step=step, t0=t == 0)
        obs, reward, done, _ = env.step(action.cpu().numpy())
        ep_reward += reward
        frame = env.render(mode='rgb_array', height=384, width=384, camera_id=0)
        writer.render(frame)
        pbar.update(1)
        t += 1
    pbar.close()

    writer.save()
    return ep_reward


score = evaluate(env, agent, 1, )
print(score)