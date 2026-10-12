"""The `atlas-harness` entry point and run labelling, without Atlas, GPU or engines."""

import contextlib
import io
import unittest
from unittest import mock

import mongomock

from atlas_gpu_harness import cli
from atlas_gpu_harness.store import AtlasStore


class CliTests(unittest.TestCase):
    def test_help_lists_every_command(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.main([])
        for name in ("doctor", "inspect", "tune", "serve", "deploy", "strata", "setup-engine", "topk", "ui"):
            self.assertIn(name, out.getvalue())

    def test_unknown_command_exits_2(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                cli.main(["nope"])
        self.assertEqual(raised.exception.code, 2)

    def test_delegate_passes_fixed_subcommand_and_prog(self):
        seen = {}

        def fake_main():
            import sys
            seen["argv"] = list(sys.argv)

        with mock.patch("importlib.import_module", return_value=mock.Mock(main=fake_main)):
            cli.main(["tune", "--model", "m.gguf"])
        self.assertEqual(seen["argv"], ["atlas-harness tune", "evolve", "--model", "m.gguf"])

    def test_doctor_reports_missing_uri_and_engine(self):
        with mock.patch.dict("os.environ", {}, clear=True), \
                mock.patch("dotenv.load_dotenv"), \
                mock.patch.object(cli, "_check_gpu", return_value=cli._row("ok", "NVIDIA GPU", "x")), \
                mock.patch.object(cli, "_check_ollama", return_value=cli._row("ok", "Ollama", "x")):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = cli.doctor(["--engine", "/nonexistent/launch-llama.sh"])
        self.assertEqual(code, 1)
        self.assertIn("MONGODB_URI is not set", out.getvalue())
        self.assertIn("atlas-harness setup-engine", out.getvalue())


class CreateRunTests(unittest.TestCase):
    def test_labels_and_extra_fields_written_in_one_insert(self):
        store = AtlasStore.__new__(AtlasStore)
        store.runs = mongomock.MongoClient().db.runs
        run_id = store.create_run({}, "ctx", 3, objective="tune MoE", policy_version="v9", port=8081)
        run = store.runs.find_one({"_id": run_id})
        self.assertEqual((run["objective"], run["policy_version"], run["port"]), ("tune MoE", "v9", 8081))
        self.assertEqual(run["status"], "running")


if __name__ == "__main__":
    unittest.main()
