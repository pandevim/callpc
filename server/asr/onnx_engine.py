#!/usr/bin/env python3
"""ONNX Runtime engine for Nemotron 3.5 ASR — torch/NeMo-free.

Loads the exported encoder.onnx (cache-aware, prompt-conditioned) + decoder_joint
.onnx and runs greedy RNNT decoding. ~0.57 GB host RAM vs ~1.7 GB for PyTorch.

This first cut does OFFLINE transcription (whole clip, empty caches) to validate
correctness; streaming chunking comes next.
"""
import json
import os

import numpy as np
import onnxruntime as ort
import sentencepiece as spm

import onnx_features as feats

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "onnx")
SEP = "▁"


def _providers():
    avail = ort.get_available_providers()
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"]


class OnnxASR:
    def __init__(self, model_dir=MODEL_DIR, lang="en-US"):
        self.cfg = json.load(open(os.path.join(model_dir, "config.json")))
        so = ort.SessionOptions()
        # Under a SLURM/cgroup cpu allocation, onnxruntime's default thread pool
        # tries to pin threads to host cores outside the allowed set
        # (pthread_setaffinity_np EINVAL flood). Setting intra-op threads
        # explicitly (e.g. ORT_INTRA_OP_THREADS=$(nproc)) stops it touching affinity.
        _nt = int(os.environ.get("ORT_INTRA_OP_THREADS", "0") or 0)
        if _nt > 0:
            so.intra_op_num_threads = _nt
            so.inter_op_num_threads = 1
        prov = _providers()
        self.enc = ort.InferenceSession(os.path.join(model_dir, "encoder.onnx"), so, providers=prov)
        self.dec = ort.InferenceSession(os.path.join(model_dir, "decoder_joint.onnx"), so, providers=prov)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(model_dir, "tokenizer.model"))
        self.blank = self.cfg["blank_id"]
        self.prompt_index = self.cfg["prompt_dictionary"].get(lang, 0)
        # decoder LSTM state shapes from the model
        self._state_shapes = {i.name: i.shape for i in self.dec.get_inputs()}

    def _empty_cache(self):
        cs = self.cfg["cache_shapes"]
        return (
            np.zeros(cs["cache_last_channel"], np.float32),
            np.zeros(cs["cache_last_time"], np.float32),
            np.zeros(cs["cache_last_channel_len"], np.int64),
        )

    def _zero_states(self):
        # input_states_1/2 : (2, batch=1, 640)
        s1 = np.zeros([2, 1, 640], np.float32)
        s2 = np.zeros([2, 1, 640], np.float32)
        return s1, s2

    def _run_encoder(self, win, clc, clt, clcl):
        """One encoder step: win (128,W) + caches -> (encoded[:,:,:len], updated caches)."""
        out = self.enc.run(None, {
            "processed_signal": win[None].astype(np.float32),
            "processed_signal_length": np.array([win.shape[1]], np.int64),
            "cache_last_channel": clc,
            "cache_last_time": clt,
            "cache_last_channel_len": clcl,
            "prompt_index": np.array([self.prompt_index], np.int64),
        })
        enc_len = int(out[1][0])
        return out[0][:, :, :enc_len], out[2], out[3], out[4]

    def _decode_frames(self, encoded, state, max_symbols=10):
        """Greedy-decode encoded frames continuing `state` (s1,s2,last). Mutates state,
        returns newly emitted token ids."""
        toks = []
        for t in range(encoded.shape[2]):
            enc_t = encoded[:, :, t:t + 1].astype(np.float32)
            emitted = 0
            while emitted < max_symbols:
                out = self.dec.run(None, {
                    "encoder_outputs": enc_t,
                    "targets": np.array([[state["last"]]], np.int32),
                    "target_length": np.array([1], np.int32),
                    "input_states_1": state["s1"],
                    "input_states_2": state["s2"],
                })
                k = int(np.argmax(out[0].reshape(-1)))
                if k == self.blank:
                    break
                toks.append(k)
                state["last"] = k
                state["s1"], state["s2"] = out[2], out[3]
                emitted += 1
        return toks

    def _greedy(self, encoded, max_symbols=10):
        state = {"s1": self._zero_states()[0], "s2": self._zero_states()[1], "last": self.blank}
        return self._decode_frames(encoded, state, max_symbols)

    def _text(self, tokens):
        toks = [t for t in tokens if t != self.blank]
        if not toks:
            return ""
        pieces = self.sp.id_to_piece(toks)
        return "".join(p.replace(SEP, " ") if p.startswith(SEP) else p for p in pieces).strip()

    # The exported encoder consumes ONE chunk per call: 65 mel frames
    # (9 lookback + 56 new) -> 7 output frames, threading the caches. We slide a
    # 65-wide window with stride 56 over the (zero-padded) mel.
    WIN = 65
    HOP_FRAMES = 56
    LOOKBACK = 9
    LOG_MEL_ZERO = -16.635

    def _encode_stream(self, mel):
        """mel: (128, F) -> encoded (1,1024,T) by chunked cache-aware encoding."""
        pad = np.full((128, self.LOOKBACK), self.LOG_MEL_ZERO, np.float32)
        mel = np.concatenate([pad, mel], axis=1)            # prepend lookback
        clc, clt, clcl = self._empty_cache()
        chunks = []
        pos = 0
        F = mel.shape[1]
        while pos + self.LOOKBACK < F:
            win = mel[:, pos:pos + self.WIN]
            if win.shape[1] < self.WIN:                     # pad final short window
                win = np.pad(win, ((0, 0), (0, self.WIN - win.shape[1])),
                             constant_values=self.LOG_MEL_ZERO)
            out = self.enc.run(None, {
                "processed_signal": win[None].astype(np.float32),
                "processed_signal_length": np.array([win.shape[1]], np.int64),
                "cache_last_channel": clc,
                "cache_last_time": clt,
                "cache_last_channel_len": clcl,
                "prompt_index": np.array([self.prompt_index], np.int64),
            })
            enc_out, enc_len = out[0], int(out[1][0])
            clc, clt, clcl = out[2], out[3], out[4]
            if enc_len > 0:
                chunks.append(enc_out[:, :, :enc_len])
            pos += self.HOP_FRAMES
        return np.concatenate(chunks, axis=2) if chunks else np.zeros((1, 1024, 0), np.float32)

    def transcribe(self, samples):
        """Offline (chunked-replay): samples float32 mono 16k -> text."""
        encoded = self._encode_stream(feats.mel(samples))
        return self._text(self._greedy(encoded))


import re as _re
_LANG_TAG = _re.compile(r"\s*<[a-z]{2,3}(?:-[A-Za-z]{2,4})?>")


class OnnxStreamingSession:
    """Real-time streaming over the ONNX engine — same interface as the PyTorch
    stream_core.StreamingSession (set_lang/reset/feed/chunk_bytes + finalize).

    feed(pcm) appends audio, recomputes mel over the utterance, and processes any
    newly-available 65-frame windows (stride 56), threading encoder caches and the
    RNNT decoder state. Returns the incremental text. Recomputing mel each call is
    cheap (numpy FFT over a few seconds) and avoids chunk-boundary artifacts.
    """

    WIN, HOP_FRAMES, LOOKBACK, LOG_MEL_ZERO = 65, 56, 9, -16.635

    def __init__(self, asr: OnnxASR, sample_rate=16000):
        self.asr = asr
        self.sample_rate = sample_rate
        self.reset()

    @property
    def chunk_bytes(self):
        return self.HOP_FRAMES * 160 * 2  # 560 ms (one encoder step) of int16 PCM

    def set_lang(self, lang):
        self.asr.prompt_index = self.asr.cfg["prompt_dictionary"].get(lang, 0)
        self.reset()

    def reset(self):
        self._audio = np.zeros(0, np.float32)
        self._clc, self._clt, self._clcl = self.asr._empty_cache()
        s1, s2 = self.asr._zero_states()
        self._state = {"s1": s1, "s2": s2, "last": self.asr.blank}
        self._pos = 0  # consumed mel-frame position (in lookback-padded coords)

    def _padded_mel(self):
        mel = feats.mel(self._audio)
        pad = np.full((128, self.LOOKBACK), self.LOG_MEL_ZERO, np.float32)
        return np.concatenate([pad, mel], axis=1)

    def _emit(self, tokens):
        toks = [t for t in tokens if t != self.asr.blank]
        if not toks:
            return ""
        pieces = self.asr.sp.id_to_piece(toks)
        text = "".join(p.replace(SEP, " ") if p.startswith(SEP) else p for p in pieces)
        return _LANG_TAG.sub("", text)

    def feed(self, pcm_int16) -> str:
        self._audio = np.concatenate([self._audio, pcm_int16.astype(np.float32) / 32768.0])
        mel = self._padded_mel()
        out = []
        while self._pos + self.WIN <= mel.shape[1]:
            win = mel[:, self._pos:self._pos + self.WIN]
            enc, self._clc, self._clt, self._clcl = self.asr._run_encoder(win, self._clc, self._clt, self._clcl)
            out.append(self._emit(self.asr._decode_frames(enc, self._state)))
            self._pos += self.HOP_FRAMES
        return "".join(out)

    def finalize(self) -> str:
        """Flush the final partial window at utterance end."""
        mel = self._padded_mel()
        if self._pos + self.LOOKBACK >= mel.shape[1]:
            return ""
        win = mel[:, self._pos:self._pos + self.WIN]
        if win.shape[1] < self.WIN:
            win = np.pad(win, ((0, 0), (0, self.WIN - win.shape[1])), constant_values=self.LOG_MEL_ZERO)
        enc, self._clc, self._clt, self._clcl = self.asr._run_encoder(win, self._clc, self._clt, self._clcl)
        self._pos += self.HOP_FRAMES
        return self._emit(self.asr._decode_frames(enc, self._state))


if __name__ == "__main__":
    import sys
    import resource
    import soundfile as sf

    wav = sys.argv[1] if len(sys.argv) > 1 else "sample.wav"
    data, sr = sf.read(wav, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    asr = OnnxASR(lang="en-US")
    print("OFFLINE:", repr(asr.transcribe(data)))

    # streaming self-test: feed in chunk_bytes-sized pieces, then finalize
    sess = OnnxStreamingSession(asr)
    pcm = (data * 32768.0).astype(np.int16)
    step = sess.chunk_bytes // 2
    parts = []
    for i in range(0, len(pcm), step):
        parts.append(sess.feed(pcm[i:i + step]))
    parts.append(sess.finalize())
    print("STREAMING:", repr("".join(parts).strip()))
    print(f"host RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")
