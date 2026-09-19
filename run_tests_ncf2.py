import subprocess, sys, os
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["TF_CPP_MIN_LOG_LEVEL"] = "3"
r = subprocess.run(
    [sys.executable, "-m", "pytest",
     "tests/test_ncf.py", "test_weekly_meal_planner.py",
     "-v", "--tb=short", "-q", "--no-header"],
    capture_output=True, text=True, encoding="utf-8",
    errors="replace", cwd=r"d:\Healthcare_Project",
    env=env, timeout=360,
)
out = r.stdout + r.stderr
print(out[-6000:] if len(out) > 6000 else out)
print("returncode:", r.returncode)
