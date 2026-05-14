import unittest

from src.runtime_settings import env_bool, get_runtime_settings, resolve_backend_choice


class RuntimeSettingsTest(unittest.TestCase):
    def test_env_bool_parsing(self):
        env = {"A": "true", "B": "0", "C": "unexpected"}
        self.assertTrue(env_bool("A", default=False, environ=env))
        self.assertFalse(env_bool("B", default=True, environ=env))
        self.assertTrue(env_bool("C", default=True, environ=env))
        self.assertFalse(env_bool("MISSING", default=False, environ=env))

    def test_get_runtime_settings_defaults(self):
        settings = get_runtime_settings(environ={})
        self.assertTrue(settings["demo_mode"])
        self.assertFalse(settings["allow_live_runs"])
        self.assertEqual(settings["default_config_path"], "evals/config.yaml")
        self.assertEqual(settings["reports_dir"], "reports")
        self.assertFalse(settings["hf_token_present"])

    def test_backend_resolution(self):
        self.assertEqual(resolve_backend_choice("clip", demo_mode=True, allow_live_runs=True), "demo")
        self.assertEqual(resolve_backend_choice("clip", demo_mode=False, allow_live_runs=False), "demo")
        self.assertEqual(resolve_backend_choice("clip", demo_mode=False, allow_live_runs=True), "clip")
        self.assertEqual(resolve_backend_choice("invalid", demo_mode=False, allow_live_runs=True), "auto")


if __name__ == "__main__":
    unittest.main()
