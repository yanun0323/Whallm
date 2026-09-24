from __future__ import annotations

import base64
import io
import unittest

import numpy as np
from PIL import Image
import mlx.core as mx

from deepseek_v4_ssd.media import (AssetStore, ImagePart, MediaError, ModalSpan, PreparedPrompt,
                                  ordered_content, media_request, MAX_ASSET_BYTES)
from deepseek_v4_ssd.mimo.processor import image_input, smart_resize, bilinear
from deepseek_v4_ssd.mimo.vision import VisionAttention, positions
from deepseek_v4_ssd.server import _messages, _response_messages
from deepseek_v4_ssd.tool_codec import ToolChoice


def png(color="red", size=(32, 64)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class MediaTests(unittest.TestCase):
    def test_asset_ownership_expiry_quota_and_cleanup(self):
        now = [0]
        store = AssetStore(ttl=10, quota=10000, max_files=1, clock=lambda: now[0])
        root = store.root
        try:
            data = png()
            item = store.put(io.BytesIO(data), len(data), "image/png", "alice")
            self.assertEqual(store.get(item["id"], "alice").data, data)
            with self.assertRaises(MediaError): store.get(item["id"], "bob")
            with self.assertRaises(MediaError): store.delete(item["id"], "bob")
            with self.assertRaises(MediaError): store.get("../../etc/passwd", "alice")
            with self.assertRaises(MediaError): store.put(io.BytesIO(data), len(data), "image/png", "alice")
            now[0] = 11
            with self.assertRaises(MediaError): store.get(item["id"], "alice")
            self.assertEqual(list(root.iterdir()), [])
            replacement = store.put(io.BytesIO(data), len(data), "image/png", "alice")
            store.delete(replacement["id"], "alice")
            with self.assertRaises(MediaError): store.get(replacement["id"], "alice")
        finally:
            store.close()
        self.assertFalse(root.exists())

    def test_upload_limits_and_truncation_leave_no_files(self):
        store = AssetStore()
        try:
            for length, mime in ((MAX_ASSET_BYTES + 1, "image/png"), (0, "image/png"),
                                 (10, "image/png"), (1, "text/html")):
                with self.subTest(length=length, mime=mime), self.assertRaises(MediaError):
                    store.put(io.BytesIO(b"x"), length, mime, "owner")
            self.assertEqual(list(store.root.iterdir()), [])
        finally:
            store.close()

    def test_ordered_parts_and_both_api_adapters_preserve_placement(self):
        data = base64.b64encode(png()).decode()
        content = [{"type": "text", "text": "before"},
                   {"type": "input_image", "image_url": "data:image/png;base64," + data},
                   {"type": "text", "text": "after"}]
        parse = lambda value, param: ordered_content(value)
        chat = _messages([{"role": "user", "content": content}], content_parser=parse)
        response = _response_messages({"input": [{"role": "user", "content": content}]}, content_parser=parse)
        self.assertEqual(chat, response)
        self.assertEqual(chat[0]["content"][0], "before")
        self.assertIsInstance(chat[0]["content"][1], ImagePart)
        self.assertEqual(chat[0]["content"][2], "after")
        self.assertIsNotNone(media_request(chat, "chat", [], ToolChoice(), "low"))
        chat[0]["role"] = "assistant"
        with self.assertRaises(MediaError): media_request(chat, "chat", [], ToolChoice(), "low")

    def test_remote_local_urls_bad_base64_and_unknown_parts_are_rejected(self):
        for url in ("https://example.com/image.png", "http://127.0.0.1/", "file:///etc/passwd",
                    "data:image/png;base64,@@@@", "data:text/html;base64,eA=="):
            with self.subTest(url=url), self.assertRaises(MediaError):
                ordered_content([{"type": "image_url", "image_url": {"url": url}}])
        for kind in ("input_audio", "video", "file"):
            with self.assertRaises(MediaError): ordered_content([{"type": kind}])

    def test_request_wide_media_count_is_bounded_across_messages(self):
        part = ImagePart.from_bytes(png(), "image/png")
        messages = [{"role": "user", "content": (part,)} for _ in range(9)]
        with self.assertRaises(MediaError): media_request(messages, "chat", [], ToolChoice(), "low")

    def test_processor_matches_normalization_and_patch_order(self):
        result = image_input(ImagePart.from_bytes(png(size=(128, 64)), "image/png"))
        self.assertEqual(result.grid, (1, 4, 8))
        self.assertEqual(result.patches.shape, (32, 1536))
        expected = (np.array([255, 0, 0]) - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
        np.testing.assert_allclose(result.patches.reshape(32, 3, 2, 16, 16)[0, :, 0, 0, 0], expected, atol=1e-6)
        with self.assertRaises(MediaError): image_input(ImagePart.from_bytes(png(), "image/jpeg"))
        with self.assertRaises(MediaError): image_input(ImagePart.from_bytes(b"bad", "image/png"))
        with self.assertRaises(MediaError): image_input(ImagePart.from_bytes(png(size=(2048, 1024)), "image/png"))

    def test_bilinear_uses_fused_float32_coordinate_rounding(self):
        image = np.arange(197 * 173 * 3, dtype=np.float32).reshape(197, 173, 3) % 256
        result = bilinear(image, 192, 160)
        # A coordinate that differs if (i+.5)*scale and subtraction round separately.
        row, col = 160, 80
        y = np.float32((row + .5) * float(np.float32(197 / 192)) - .5)
        x = np.float32((col + .5) * float(np.float32(173 / 160)) - .5)
        y0, x0 = int(y), int(x)
        expected = sum(image[y0 + dy, x0 + dx] * np.float32((1 - abs(float(y) - y0 - dy)) *
                              (1 - abs(float(x) - x0 - dx))) for dy in (0, 1) for dx in (0, 1))
        np.testing.assert_allclose(result[row, col], expected, atol=3e-5)
        self.assertEqual(smart_resize(197, 173), (192, 160))

    def test_prepared_span_alignment_is_checked(self):
        span = ModalSpan(1, 1, "digest", (1, 2, 2))
        PreparedPrompt((1, 2, 3), np.zeros((3, 4)), (span,), "id").validate(4)
        for span, shape in ((ModalSpan(3, 1, "d", (1, 2, 2)), (3, 4)), (span, (2, 4))):
            with self.assertRaises(MediaError):
                PreparedPrompt((1, 2, 3), np.zeros(shape), (span,), "id").validate(4)

    def test_vision_attention_uses_denominator_sink_not_first_key_bias(self):
        config = {"num_heads": 4, "num_key_value_heads": 2, "qk_channels": 16,
                  "hidden_size": 32, "use_sink": True, "visual_token_window_size": 2}
        mx.random.seed(4)
        module = VisionAttention(config, False)
        module.sinks = mx.array([0., 1., -1., 2.])
        x = mx.random.normal((16, 32))
        actual = np.asarray(module(x, mx.zeros((16, 16)), (1, 4, 4)))
        q, k, v = np.split(np.asarray(module.qkv(x)), [64, 96], axis=-1)
        q = q.reshape(16, 4, 16).transpose(1, 0, 2)
        k, v = (np.repeat(a.reshape(16, 2, 16).transpose(1, 0, 2), 2, 0) for a in (k, v))
        scores = q @ k.transpose(0, 2, 1) / 4
        pos = np.arange(16)
        scores = np.where(np.abs(pos[:, None] - pos[None, :]) <= 2, scores, -np.inf)
        scores = np.concatenate((scores, np.broadcast_to(np.array([0., 1., -1., 2.])[:, None, None], (4, 16, 1))), axis=-1)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True)); probs /= probs.sum(axis=-1, keepdims=True)
        out = (probs[..., :-1] @ v).transpose(1, 0, 2).reshape(16, 64)
        expected = out @ np.asarray(module.proj.weight).T + np.asarray(module.proj.bias)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        coords, columns = positions((1, 4, 6))
        np.testing.assert_array_equal(np.asarray(columns), [0, 3, 1, 4, 2, 5])
        np.testing.assert_array_equal(np.asarray(coords)[:4], [[0, 0], [0, 1], [1, 0], [1, 1]])
