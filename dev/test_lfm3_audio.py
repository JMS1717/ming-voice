"""Offline checks of the experimental adapter: python dev/test_lfm3_audio.py."""
import sys
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lfm3_audio import Lfm3Audio, _CancelStreamer, _GenerationCancelled  # noqa: E402


class TestLfm3Audio(unittest.TestCase):
    def setUp(self):
        self.backend = Lfm3Audio.__new__(Lfm3Audio)
        self.backend.max_ctx = 4096
        self.backend.tok = Mock()
        self.backend.tok.encode.side_effect = lambda text, **_: text.split()
        self.backend.reset("Be helpful.")
        self.backend._generate = Mock()

    def test_spoken_chat_and_rollback(self):
        self.backend._generate.return_value = ("Hello there!<|im_end|>", 32, None)
        wav = np.zeros(16000, np.float32)
        snap = self.backend.snapshot()
        answer = list(self.backend.stream(wav))
        self.assertEqual(answer, ["Hello there!"])
        args, kwargs = self.backend._generate.call_args
        self.assertIs(kwargs["audio"], wav)
        self.assertEqual(args[0][-1]["content"], [{"type": "audio"}])
        self.assertEqual(self.backend.ctx_len(), 32)
        self.backend.restore(snap)
        self.assertEqual(self.backend.history, [])
        self.assertEqual(self.backend.ctx_len(), 0)

    def test_tool_response_and_speculation_gate(self):
        toolbox = Mock()
        toolbox.home = ""
        toolbox.names = ["timer"]
        toolbox.resolve.return_value = ("timer", {})
        toolbox.run.return_value = ("Set.", {})
        self.backend.set_tools(toolbox)
        call = "<tool_call><function=timer><parameter=when>2 minutes</parameter></function></tool_call>"
        with patch("lfm3_audio.tool_guidance", return_value=""):
            self.backend._generate.return_value = (call, 32, None)
            self.assertEqual(list(self.backend.stream(None, user_text="Set a timer", gate=lambda: False)), [])
            toolbox.run.assert_not_called()
            self.backend._generate.reset_mock()
            self.backend._generate.side_effect = [(call, 32, None), ("Timer set.", 64, None)]
            items = list(self.backend.stream(None, user_text="Set a timer", events=True, gate=lambda: True))
        self.assertEqual([item["type"] for item in items[:-1]], ["tool_call", "tool_result"])
        self.assertEqual(items[-1], "Timer set.")
        self.assertEqual(toolbox.run.call_args.args[:2], ("timer", {"when": "2 minutes"}))
        messages = self.backend._generate.call_args.args[0]
        self.assertEqual(messages[-1], {"role": "tool", "content": '["Set."]'})

    def test_api_chat_history_isolated(self):
        self.backend._generate.return_value = ("Answer", 30, None)
        self.backend.history = [{"user": "Live", "content": "Turn"}]
        with self.backend.isolated():
            self.backend.reset("API")
            list(self.backend.stream(None, user_text="Now", history_turns=[("Earlier", "Reply")]))
            messages = self.backend._generate.call_args.args[0]
            self.assertEqual([message["content"] for message in messages], ["API", "Earlier", "Reply", "Now"])
        self.assertEqual(self.backend.history, [{"user": "Live", "content": "Turn"}])

    def test_synthesis_uses_locale_and_cancel_streamer(self):
        pcm = np.ones(2400, dtype=np.float32)
        self.backend._generate.return_value = ("", 10, pcm)
        chunks = list(self.backend.synth_stream("Say this.", "UK_George"))
        self.assertIs(chunks[0], pcm)
        args, kwargs = self.backend._generate.call_args
        self.assertIn("British English using a male voice", args[0][0]["content"])
        self.assertTrue(kwargs["speak"])
        self.assertEqual(self.backend.SR, 24000)
        cancel = threading.Event()
        streamer = _CancelStreamer(cancel)
        streamer.put(None)
        cancel.set()
        with self.assertRaises(_GenerationCancelled):
            streamer.put(None)

    def test_processor_input_and_context_limit(self):
        self.backend._generate = Lfm3Audio._generate.__get__(self.backend)
        self.backend.lock = threading.Lock()
        self.backend.torch = SimpleNamespace(inference_mode=nullcontext)
        self.backend.device = "cuda:0"
        inputs = Mock()
        inputs.__getitem__ = Mock(return_value=np.empty((1, 4093)))
        inputs.to.return_value = {"input_ids": np.empty((1, 4093))}
        self.backend.processor = Mock(return_value=inputs)
        self.backend.processor.apply_chat_template.return_value = "user message with audio placeholder"
        self.backend.model = Mock()
        self.backend.model.generate.return_value = SimpleNamespace(
            sequences=np.arange(4096).reshape(1, -1), sequence_lengths=[4096])
        self.backend.tok.decode.return_value = "Transcript"
        wav = np.zeros(16000, np.float32)
        text, count, _ = self.backend._generate([], threading.Event(), 100, audio=wav)
        self.assertEqual((text, count), ("Transcript", 4096))
        self.assertIs(self.backend.processor.call_args.kwargs["audio"], wav)
        self.assertEqual(self.backend.processor.call_args.kwargs["sampling_rate"], 16000)
        self.assertEqual(self.backend.model.generate.call_args.kwargs["max_new_tokens"], 3)
        self.assertTrue(self.backend.model.generate.call_args.kwargs["suppress_audio_out"])


if __name__ == "__main__":
    unittest.main()
