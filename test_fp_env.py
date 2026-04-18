from fp_gym_env import FeasibilityPumpRLEnv, FPGymConfig
from mmp_fp_core import FPRunConfig

instance_paths = [
    "instances_n3000/instance_1.npz",
    "instances_n3000/instance_2.npz",
    "instances_n3000/instance_3.npz",
]

cfg = FPGymConfig(
    instance_paths=instance_paths,
    fp_config=FPRunConfig(
        max_iterations=100,
        time_limit=60.0,
        stall_threshold=3,
        max_stalls=50,
        cplex_threads=1,
    ),
    max_reset_resamples=20,
    seed=10,
)

env = FeasibilityPumpRLEnv(cfg)

obs, info = env.reset()
print("RESET INFO:")
print(info)
print()

print("OBS KEYS:", obs.keys())
print("progress shape:", obs["progress"].shape)
print("history shape:", obs["history"].shape)
print("instance shape:", obs["instance"].shape)
print()

# Try one manual action: flip bin 3, continuation bin 2
action = [3, 2]
obs, reward, terminated, truncated, info = env.step(action)

print("STEP RESULT:")
print("reward =", reward)
print("terminated =", terminated)
print("truncated =", truncated)
print("info =", info)