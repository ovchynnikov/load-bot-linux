"""Access control parsing fails closed; startup parsing tolerates blank list items."""

import importlib
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import permissions


class AccessParsingTests(unittest.TestCase):
    def test_only_explicit_false_values_disable_limits(self):
        for value in ('False', 'false', ' FALSE ', '0', 'no', 'off'):
            with self.subTest(value=value):
                self.assertFalse(permissions.is_access_limited(value))
        for value in ('True', 'true', "'True'", "'False'", "'False", 'False"', 'garbage', '', 'False  # comment'):
            with self.subTest(value=value):
                self.assertTrue(permissions.is_access_limited(value))

    def test_module_state_from_environment(self):
        env = {'LIMIT_BOT_ACCESS': 'false', 'ALLOWED_CHAT_IDS': ' , 42, ', 'ALLOWED_USERNAMES': ' alice , '}
        try:
            with patch.dict(os.environ, env):
                module = importlib.reload(permissions)
                self.assertFalse(module.is_user_or_chat_not_allowed('anyone', 1))
                self.assertEqual(module.allowed_chat_ids, [42])
                self.assertEqual(module.allowed_usernames, ['alice'])
            with patch.dict(os.environ, {**env, 'LIMIT_BOT_ACCESS': 'True'}):
                module = importlib.reload(permissions)
                self.assertTrue(module.is_user_or_chat_not_allowed('anyone', 1))
                self.assertFalse(module.is_user_or_chat_not_allowed('alice', 1))
                self.assertFalse(module.is_user_or_chat_not_allowed('anyone', 42))
        finally:
            importlib.reload(permissions)


class ResponsesFallbackTests(unittest.TestCase):
    def test_missing_responses_file_gives_whole_sentence(self):
        root = Path(__file__).resolve().parents[1]
        script = 'import main, random; assert isinstance(main.responses, list); '
        script += 'assert len(random.choice(main.responses)) > 10; main.db_storage.close()'
        with TemporaryDirectory() as temp:
            result = subprocess.run(
                [sys.executable, '-c', script],
                cwd=temp,
                env={
                    'PATH': os.environ.get('PATH', ''),
                    'PYTHONPATH': str(root / 'src'),
                    'PYTHON_DOTENV_DISABLED': '1',
                },
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
