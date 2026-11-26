"""Exercise the real service commands with four disposable spill roots and a local SSH stand-in."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class ForgetTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.home = self.root / "service"
        self.run = self.home / "run"
        self.run.mkdir(parents=True)
        self.script = Path(os.environ.get("SERVICE_UNDER_TEST", Path(__file__).resolve().parents[1] / "service.sh"))
        self.table = self.run / "front.table"
        self.table.touch()
        fleet = self.root / "fleet.conf"
        self.ranks = [self.root / f"spill{rank}" for rank in range(4)]
        fleet.write_text("\n".join(["BIN_DIR=bin", "LOG_DIR=log", f"SERVICE_DIR={self.home}",
                                    f"RING_LOCK={self.root}/ring.lock",
                                    *(f"RANK{i}_SSH=worker{i}" for i in range(1, 4)),
                                    *(f"RANK{i}_SPILL={path}" for i, path in enumerate(self.ranks))]))
        ssh = self.root / "ssh"
        ssh.write_text('#!/bin/bash\nwhile [[ $1 == -o ]]; do shift 2; done\n'
                       '[[ $1 =~ ^worker[123]$ ]] || exit 96\n'
                       '[[ $1 != "${FAIL_WORKER:-}" ]] || exit 255\n'
                       'shift; exec bash -c "$1"\n')
        ssh.chmod(0o700)
        self.env = {**os.environ, "FLEET": str(fleet), "DSGB10_HOME": str(self.home),
                    "DSGB10_MOCK": "0", "PATH": f"{self.root}:{os.environ['PATH']}"}
        self.image = "ordinary-image.service." + "a" * 16 + "-" + "b" * 64
        self.other = self.image.replace(".service.", ".trial.")
        for i, path in enumerate(self.ranks):
            path.mkdir()
            (path / f"rank{i}.service.spill").touch()
            (path / self.image / f"rank{i}").mkdir(parents=True)
            (path / self.image / f"rank{i}" / "page").touch()
            (path / self.other).mkdir()
            (path / "ordinary-image.service.keep").mkdir()
            (path / "unrelated.service.data").touch()

    def service(self, *args, ok=True, **env):
        p = subprocess.run(["bash", str(self.script), *args], env={**self.env, **env},
                           capture_output=True, text=True, timeout=10)
        self.assertEqual(p.returncode == 0, ok, p.stderr + p.stdout)

    def test_stop_preserves_then_explicit_forget_is_idempotent(self):
        self.service("stop")
        self.assertTrue(self.table.exists())
        for path in self.ranks:
            self.assertTrue((path / self.image).is_dir())
        self.service("stop", "--forget")
        self.service("forget")
        self.assertFalse(self.table.exists())
        for i, path in enumerate(self.ranks):
            self.assertFalse((path / f"rank{i}.service.spill").exists())
            self.assertFalse((path / self.image).exists())
            self.assertTrue((path / self.other).is_dir())
            self.assertTrue((path / "ordinary-image.service.keep").is_dir())
            self.assertTrue((path / "unrelated.service.data").is_file())

    def test_rank_failure_is_retryable(self):
        self.service("forget", ok=False, FAIL_WORKER="worker2")
        self.assertTrue((self.ranks[2] / self.image).is_dir())
        self.service("forget")
        self.assertFalse((self.ranks[2] / self.image).exists())

    def test_locked_supervisor_and_mock_protect_spill(self):
        import fcntl
        with (self.run / "supervisor.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.service("forget", ok=False)
            self.assertTrue(self.table.exists())
        self.service("forget", DSGB10_MOCK="1")
        self.assertTrue(all((p / self.image).exists() for p in self.ranks))

    def test_orphan_frontend_refuses_before_removal(self):
        (self.run / "front.pid").write_text(str(os.getpid()))
        self.service("forget", ok=False)
        self.assertTrue(self.table.exists())
        self.assertTrue(all((p / self.image).is_dir() for p in self.ranks))

    def test_use_frozen_spill_paths(self):
        config = (self.root / "fleet.conf").read_text()
        (self.home / "bin").mkdir()
        (self.home / "bin/fleet.conf").write_text(config)
        (self.root / "fleet.conf").write_text(config.replace("/spill", "/other"))
        self.service("forget")
        self.assertFalse(any((p / self.image).exists() for p in self.ranks))

    def test_symlink_is_refused(self):
        alias = self.ranks[0] / self.image.replace("a" * 16, "d" * 16)
        alias.symlink_to(self.ranks[0] / self.other, target_is_directory=True)
        self.service("forget", ok=False)
        self.assertTrue(alias.is_symlink())
        self.assertTrue((self.ranks[0] / self.other).is_dir())


if __name__ == "__main__":
    unittest.main()
