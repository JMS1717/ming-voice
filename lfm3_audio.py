"""Experimental local speech backend using one LFM3-Audio model for chat, ASR and TTS."""
from __future__ import annotations

import json
import os
import re
import textwrap
import threading
from contextlib import contextmanager

import numpy as np

from tools import SIDE_EFFECTS, get_current_time, tool_guidance

MODEL_ID = "LiquidAI/LFM3-Audio-2B"
REVISION = "2d6c1f99c129b3ef07a8d168bc0ff08e2bdb93f1"
VOICES = {
    "US_Stella": ("American English", "female"),
    "US_Emma": ("American English", "female"),
    "US_James": ("American English", "male"),
    "UK_Charlotte": ("British English", "female"),
    "UK_George": ("British English", "male"),
    "灵小柔": ("Chinese", "female"),
    "灵楷": ("Chinese", "male"),
}
_CALL = re.compile(r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", re.S)
_ARG = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.S)
_SPECIAL = re.compile(r"<\|[^<>]+\|>")
_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.S)


def _tool_calls(text):
    for name, body in _CALL.findall(text):
        args = {}
        for key, value in _ARG.findall(body):
            try:
                args[key] = json.loads(value)
            except ValueError:
                args[key] = value
        yield {"type": "function", "function": {"name": name, "arguments": args}}


def _user_message(text="", audio=False):
    if not audio:
        return {"role": "user", "content": text}
    parts = [{"type": "audio"}]
    if text:
        parts.append({"type": "text", "text": text})
    return {"role": "user", "content": parts}


class _GenerationCancelled(Exception):
    pass


class _CancelStreamer:
    # LFM3's generate does not accept stopping_criteria. Its streamers see each token or audio frame.
    def __init__(self, cancel):
        self.cancel = cancel

    def put(self, _token):
        if self.cancel.is_set():
            raise _GenerationCancelled

    def end(self):
        pass


class Lfm3Audio:
    SR = 24000

    def __init__(self, system_prompt):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        self.device = "cuda:0"
        local_dir = os.environ.get("MING_LFM3_MODEL_DIR")
        source = os.path.expanduser(local_dir) if local_dir else MODEL_ID
        opts = {"local_files_only": True} if local_dir else {"revision": REVISION}
        self.processor = AutoProcessor.from_pretrained(source, trust_remote_code=True, **opts)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            source, dtype=torch.bfloat16, device_map=self.device, trust_remote_code=True, **opts).eval()
        self.codec = self.processor.audio_tokenizer.to(device=self.device).eval()
        self.tok = self.processor.tokenizer
        self.max_ctx = int(os.environ.get("MING_MAX_CTX", "4096"))
        if not 2048 <= self.max_ctx <= 32768:
            raise ValueError("LFM3-Audio requires MING_MAX_CTX between 2048 and 32768")
        # The server can run synthesis on a different thread while processing a tool call.
        self.lock = threading.Lock()
        self.reset(system_prompt)

    def reset(self, system_prompt=None):
        if system_prompt is not None:
            self.system_prompt = system_prompt
        self.history = []
        self.cur_len = 0
        self.tools = None

    def set_tools(self, tools):
        self.tools = tools

    def ctx_len(self):
        return self.cur_len

    def needs_compact(self):
        return False  # each generation rebuilds a bounded context from the transcript

    def snapshot(self):
        return self.cur_len, len(self.history)

    def restore(self, snap):
        self.cur_len, n_hist = snap
        del self.history[n_hist:]

    @contextmanager
    def isolated(self):
        saved = self.system_prompt, self.history, self.cur_len, self.tools
        try:
            yield self
        finally:
            self.system_prompt, self.history, self.cur_len, self.tools = saved

    def _history_messages(self, turns):
        budget = self.max_ctx // 3
        messages = []
        for user, answer in reversed(turns[-16:]):
            cost = len(self.tok.encode(user + "\n" + answer, add_special_tokens=False)) + 12
            if cost > budget:
                break
            messages[:0] = [_user_message(user), {"role": "assistant", "content": answer}]
            budget -= cost
        return messages

    def _generate(self, messages, cancel, max_tokens, *, audio=None, tools=None, prefill=None, speak=False):
        """Generate one bounded continuation. Return decoded text, token count and optional 24 kHz PCM."""
        template_messages = messages
        if prefill:
            template_messages = messages + [{"role": "assistant", "content": prefill}]
        with self.lock, self.torch.inference_mode():
            if cancel.is_set():
                raise _GenerationCancelled
            if audio is None:
                inputs = self.processor.apply_chat_template(
                    template_messages, tools=tools, tokenize=True, return_dict=True, return_tensors="pt",
                    add_generation_prompt=not prefill, continue_final_message=bool(prefill))
            else:
                prompt = self.processor.apply_chat_template(
                    template_messages, tools=tools, tokenize=False, add_generation_prompt=not prefill,
                    continue_final_message=bool(prefill))
                inputs = self.processor(text=prompt, audio=audio, sampling_rate=16000, return_tensors="pt")
            start = inputs["input_ids"].shape[-1]
            limit = min(max_tokens, self.max_ctx - start)
            if limit <= 0:
                raise ValueError("The input is too long for the LFM3-Audio context")
            streamer = _CancelStreamer(cancel)
            options = (dict(audio_temperature=1.0, audio_top_k=128, audio_repetition_penalty=1.1,
                            audio_repetition_window=4096, audio_streamer=streamer) if speak
                       else dict(suppress_mixed=True, suppress_audio_out=True))
            output = self.model.generate(**inputs.to(self.device), max_new_tokens=limit, do_sample=False,
                                         return_dict_in_generate=True, streamer=streamer, **options)
            if cancel.is_set():
                raise _GenerationCancelled
            if speak:
                length = int(output.audio_lengths[0])
                if not length:
                    return "", start, np.zeros(0, np.float32)
                pcm = self.codec.decode(output.audio_codes[:1, :, :length]).audio_values[0]
                return "", start, pcm.float().cpu().numpy().reshape(-1)
            end = int(output.sequence_lengths[0])
            text = self.tok.decode(output.sequences[0, start:end], skip_special_tokens=False)
            return text, end, None

    def transcribe(self, wav16k, max_new_tokens=200, cancel=None):
        cancel = cancel or threading.Event()
        msgs = [{"role": "system", "content": "Perform ASR."}, _user_message(audio=True)]
        pieces = []
        try:
            for offset in range(0, len(wav16k), 30 * 16000):
                if max_new_tokens <= 0:
                    break
                text, _, _ = self._generate(msgs, cancel, min(256, max_new_tokens),
                                            audio=wav16k[offset:offset + 30 * 16000])
                text = _SPECIAL.sub("", text).strip()
                max_new_tokens -= len(self.tok.encode(text, add_special_tokens=False))
                pieces.append(text)
        except _GenerationCancelled:
            return ""
        return " ".join(filter(None, pieces))

    def synth_stream(self, text, voice="US_Stella", cancel=None):
        cancel = cancel or threading.Event()
        locale, gender = VOICES.get(voice, VOICES["US_Stella"])
        system = {"role": "system", "content": f"Perform TTS in {locale} using a {gender} voice."}
        try:
            for piece in textwrap.wrap(text, width=240):
                _, _, pcm = self._generate([system, _user_message(piece)], cancel, 384,
                                           prefill="<|audio_start|>", speak=True)
                if pcm.size:
                    yield pcm
        except _GenerationCancelled:
            return

    def synthesize(self, text, voice):
        chunks = list(self.synth_stream(text, voice))
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    def stream(self, wav16k, cancel=None, max_new_tokens=192, user_text=None, history_turns=None,
               events=False, gate=None):
        cancel = cancel or threading.Event()
        heard = {}
        toolbox = self.tools

        def said():
            if "text" not in heard:
                spoken = self.transcribe(wav16k, max_new_tokens=100, cancel=cancel) if wav16k is not None else ""
                heard["text"] = (spoken + " " + (user_text or "")).strip()
            return heard["text"]

        system = self.system_prompt
        if toolbox is not None:
            toolbox.new_turn(said)
            system += f"\nThe local date and time is {get_current_time()[0]}.\n{tool_guidance(toolbox.home)}\nList of tools:"
        turns = history_turns if history_turns is not None else [
            (h.get("user") or "(inaudible)", h.get("content") or "") for h in self.history[-16:]]
        history = self._history_messages(turns)
        messages = [{"role": "system", "content": system}, *history,
                    _user_message(user_text or "", audio=wav16k is not None)]
        reply = ""
        seen = set()
        try:
            for round_no in range(4):
                text, self.cur_len, _ = self._generate(
                    messages, cancel, max(1, min(max_new_tokens, 512)), audio=wav16k,
                    tools=toolbox.specs if toolbox else None,
                    prefill=None if toolbox else "<think></think>")
                text = _THINK.sub("", _SPECIAL.sub("", text)).replace("</think>", "")
                calls = []
                for call in _tool_calls(text):
                    fn = call["function"]
                    identity = json.dumps(fn, sort_keys=True)
                    if toolbox and fn["name"] in toolbox.names and identity not in seen:
                        seen.add(identity)
                        calls.append(call)
                    if len(calls) == 3:
                        break
                if not calls or round_no == 3:
                    reply = text.split("<tool_call>", 1)[0].strip()
                    break
                results, says = [], []
                for call in calls:
                    name, args = call["function"]["name"], call["function"]["arguments"]
                    if gate is not None and toolbox.resolve(name, args)[0] in SIDE_EFFECTS and not gate():
                        return
                    if cancel.is_set():
                        return
                    if events:
                        yield {"type": "tool_call", "name": name, "args": args}
                    result, info = toolbox.run(name, args, cancel)
                    if events:
                        yield {"type": "tool_result", "name": name, "args": args, **info}
                    results.append(result)
                    says.append(info.get("say"))
                if all(says):
                    reply = " ".join(says)
                    break
                messages.extend([{"role": "assistant", "tool_calls": calls},
                                 {"role": "tool", "content": json.dumps(results)}])
            if not cancel.is_set():
                reply = reply or "I couldn't complete that request."
                yield reply
        except _GenerationCancelled:
            pass
        finally:
            self.history.append({"role": "assistant", "content": reply, "interrupted": cancel.is_set(),
                                 "user": user_text if user_text is not None else heard.get("text")})
