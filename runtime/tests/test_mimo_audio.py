import base64
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import mlx.core as mx
import numpy as np

from deepseek_v4_ssd.cancellation import cancellation_scope, GenerationCancelled
from deepseek_v4_ssd.media import (AudioPart, AssetStore, MediaError, MediaRequest, ordered_content,
                                  media_request, expand_documents, DocumentPart)
from deepseek_v4_ssd.mimo.audio_processor import decode_wav, log_mel, audio_input
from deepseek_v4_ssd.mimo.audio import AudioTokenizer, AudioPatchEncoder, TokenizerAttention
from deepseek_v4_ssd.mimo.inputs import validate_media, prepare
from deepseek_v4_ssd.documents import DOCX
from runtime.tests.test_documents import docx


def wav(samples=2400, rate=24000, width=2, channels=1):
    output = io.BytesIO()
    with wave.open(output, "wb") as file:
        file.setparams((channels, width, rate, 0, "NONE", ""))
        file.writeframes(bytes(samples * channels * width))
    return output.getvalue()


def config():
    return {"kernel_size": 3, "stride_size": 2, "avg_pooler": 2, "activation_function": "gelu",
            "ln_type": "LayerNorm", "position_embedding_type": "rope", "rope_type": "default",
            "hybrid_attention": True, "swa_per_block": 2, "encoder_attn_window_size": [128, 0],
            "encoder_skip_layer_id": 3, "d_model": 16, "encoder_attention_heads": 2,
            "encoder_ffn_dim": 32, "encoder_layers": 4, "n_mels": 128,
            "num_quantizers": 2, "codebook_size": [4, 4], "rope_theta": 10000}


class AudioTests(unittest.TestCase):
    def test_pcm_normalization_and_stereo_mix(self):
        data = bytearray(wav(600, channels=2))
        values = np.tile(np.array([32767, -16384], dtype='<i2'), 600)
        data[-values.nbytes:] = values.tobytes()
        np.testing.assert_array_equal(decode_wav(bytes(data)), np.full(600, (32767-16384)/65536, np.float32))

    def test_duration_encoding_and_header_limits_fail_closed(self):
        for data in (wav(480), wav(720001), wav(rate=16000), wav(width=3), wav(channels=3),
                     b'not audio', wav()[:-1], wav() + b'extra'):
            with self.subTest(size=len(data)), self.assertRaises(MediaError): decode_wav(data)
        self.assertEqual(decode_wav(wav(720000)).size, 720000)

    def test_duplicate_truncated_and_partial_sample_chunks_rejected(self):
        data = wav()
        for extra in (data[12:36], data[36:], b'junk\x04\x00\x00\x00a'):
            corrupt = data + extra
            corrupt = corrupt[:4] + (len(corrupt)-8).to_bytes(4, 'little') + corrupt[8:]
            with self.assertRaises(MediaError): decode_wav(corrupt)
        # Odd data size is not silently rounded down to a whole sample.
        corrupt = bytearray(wav())
        corrupt[40:44] = (4801).to_bytes(4, 'little')
        corrupt += b'\x00\x00'
        corrupt[4:8] = (len(corrupt)-8).to_bytes(4, 'little')
        with self.assertRaises(MediaError): decode_wav(bytes(corrupt))

    def test_silence_mel_floor_and_frame_count(self):
        mel = log_mel(np.zeros(24000, np.float32))
        self.assertEqual(mel.shape, (101, 128))
        np.testing.assert_array_equal(mel, np.full((101, 128), np.log(np.float32(1e-7)), np.float32))
        inp = audio_input(AudioPart.from_bytes(wav(24000), 'audio/wav'))
        self.assertEqual(inp.token_count, 7)
        with self.assertRaises(ValueError): log_mel(np.full(1000, np.nan))

    def test_inline_and_asset_order_ownership_and_type_checks(self):
        data = wav()
        inline = {'type':'input_audio', 'input_audio': {'format':'wav', 'data':base64.b64encode(data).decode()}}
        parts = ordered_content([{'type':'text','text':'before'}, inline, {'type':'text','text':'after'}])
        self.assertEqual((parts[0], parts[2]), ('before', 'after'))
        self.assertEqual(parts[1].data, data)
        store = AssetStore()
        try:
            item = store.put(io.BytesIO(data), len(data), 'audio/wav', 'owner')
            ref = {'type':'input_audio', 'file_id':item['id']}
            self.assertEqual(ordered_content([ref], asset_store=store, owner='owner')[0], parts[1])
            for part, owner in ((ref,'other'), ({**ref,'type':'image'},'owner'), ({**ref,'type':'file'},'owner')):
                with self.assertRaises(MediaError): ordered_content([part], asset_store=store, owner=owner)
            store.delete(item['id'], 'owner')
            with self.assertRaises(MediaError): ordered_content([ref], asset_store=store, owner='owner')
        finally: store.close()

    def test_urls_formats_base64_roles_and_request_counts_rejected(self):
        for item in ({'type':'input_audio','url':'https://example.com/private.wav'},
                     {'type':'input_audio','input_audio':{'format':'mp3','data':'AAAA'}},
                     {'type':'input_audio','input_audio':{'format':'wav','data':'***'}}):
            with self.assertRaises(MediaError): ordered_content([item])
        part = AudioPart.from_bytes(wav(), 'audio/wav')
        for messages in ([{'role':'system','content':(part,)}], [{'role':'user','content':(part,)}]*3):
            with self.assertRaises(MediaError): media_request(messages, 'chat', [], None, 'low')
        request = media_request([{'role':'tool','content':('a', part)}], 'chat', [], None, 'low')
        self.assertEqual(validate_media(request).messages[0]['content'][1].token_count, 1)

    def test_images_and_audio_share_the_aggregate_media_token_budget(self):
        from deepseek_v4_ssd.media import ImagePart
        from runtime.tests.test_media import png
        image = ImagePart.from_bytes(png(), 'image/png')
        audio = AudioPart.from_bytes(wav(), 'audio/wav')
        request = media_request([{'role':'user','content':(image,audio)}], 'chat', [], None, 'low')
        with patch('deepseek_v4_ssd.mimo.inputs.image_input', return_value=SimpleNamespace(grid=(1,64,128))):
            with self.assertRaisesRegex(MediaError, '2048'): validate_media(request)

    def test_documents_and_audio_keep_order_during_expansion(self):
        audio = AudioPart.from_bytes(wav(), 'audio/wav')
        document = DocumentPart.from_bytes(docx(), DOCX)
        messages = expand_documents([{'role':'user','content':('before', document, audio, 'after')}], enabled=True)
        parts = messages[0]['content']
        self.assertIsInstance(parts, tuple)
        self.assertIn('Invoice total: 42', ''.join(p for p in parts if isinstance(p, str)))
        self.assertEqual(parts[-2:], (audio, 'after'))
        self.assertIsNotNone(media_request(messages, 'chat', [], None, 'low'))

    def test_encoder_lengths_skip_connection_and_cancellation(self):
        encoder = AudioTokenizer(config())
        mx.random.seed(7)
        mel = mx.random.normal((19, 128))
        with_skip = encoder.features(mel)
        self.assertEqual(with_skip.shape, (5, 16))
        encoder.skip_layer = None
        without = encoder.features(mel)
        self.assertFalse(mx.allclose(with_skip, without).item())
        self.assertEqual(encoder(mel).shape, (5, 2))
        event = threading.Event(); event.set()
        with cancellation_scope(event), self.assertRaises(GenerationCancelled): encoder(mel)
        with cancellation_scope(event), self.assertRaises(GenerationCancelled): log_mel(np.zeros(1000))
        with self.assertRaises(ValueError): encoder.features(mx.zeros((3002,128)))

    def test_local_attention_blocks_future_but_global_does_not(self):
        attention = TokenizerAttention(config())
        x = mx.random.normal((1, 4, 16))
        changed = mx.concatenate((x[:,:1], x[:,1:] + 10), axis=1)
        pos = mx.arange(4)
        mask = pos[None,:] <= pos[:,None]
        np.testing.assert_allclose(np.array(attention(x, mask)[:,0]), np.array(attention(changed, mask)[:,0]), atol=1e-5)
        self.assertFalse(mx.allclose(attention(x, None)[:,0], attention(changed, None)[:,0]).item())

    def test_rvq_residual_and_first_index_tie_break(self):
        encoder = AudioTokenizer(config())
        encoder.codebooks = [mx.array([[0.,0.],[1.,0.],[1.,0.]]), mx.array([[0.,0.],[0.,1.]])]
        codes = encoder.quantize(mx.array([[1.,1.],[0.,0.]]))
        np.testing.assert_array_equal(np.asarray(codes), [[1,1],[0,0]])

    def test_patch_repeats_last_code_and_attention_is_group_local(self):
        cfg = {'input_full_attention':True, 'add_post_norm':True, 'projection_layers':2,
               'partial_rotary_factor':1., 'input_local_head_dim':8, 'input_local_attn_heads':2,
               'input_local_dim':16, 'audio_channels':2, 'group_size':4, 'speech_vocab_size':8,
               'input_local_layers':2,'input_local_intermediate_size':32,'rope_theta':640000,'out_hidden_size':12}
        patch_encoder = AudioPatchEncoder(cfg)
        codes = mx.array([[1,2],[3,4],[5,6],[7,0],[2,1]])
        padded = mx.concatenate((codes, mx.repeat(codes[-1:], 3, axis=0)))
        np.testing.assert_allclose(np.asarray(patch_encoder(codes)), np.asarray(patch_encoder(padded)), atol=1e-6)
        np.testing.assert_allclose(np.asarray(patch_encoder(codes[:4])), np.asarray(patch_encoder(codes)[:1]), atol=1e-5)

    def test_audio_preparation_uses_native_tokens_and_digest_not_vision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'manifest.json').write_text('{}')
            (root/'config.json').write_text(json.dumps({'processor_config':
                {'audio_start_token_id':151673,'audio_token_id':151669,'audio_end_token_id':151674}}))
            runtime = SimpleNamespace(installed=SimpleNamespace(root=root, maximum_context=500, revision='fixture'),
                model=SimpleNamespace(args=SimpleNamespace(hidden_size=4), model=SimpleNamespace(embed_tokens=lambda t:mx.zeros((len(t),4)))),
                tokenizer=SimpleNamespace(encode=lambda s, **kw:[ord(c) for c in s]),
                _codec=SimpleNamespace(encode=lambda messages,*args:''.join(m['content'] for m in messages)))
            inp = audio_input(AudioPart.from_bytes(wav(), 'audio/wav'))
            request = MediaRequest(({'role':'user','content':('before',inp,'after')},),'chat',(),None,'low')
            with patch('deepseek_v4_ssd.mimo.vision.VisionEncoder', side_effect=AssertionError('audio must not load ViT')), \
                 patch('deepseek_v4_ssd.mimo.audio.load_audio', return_value=(lambda m:mx.zeros((3,20), mx.int32), lambda c:mx.ones((1,4)))):
                first = prepare(runtime, request)
                second = prepare(runtime, replace(request, messages=({'role':'user','content':('before',replace(inp,digest='different'),'after')},)))
            self.assertEqual(first.token_ids, tuple(map(ord,'before')) + (151673,151669,151674) + tuple(map(ord,'after')))
            self.assertEqual(first.spans[0].kind, 'audio')
            self.assertEqual(first.token_ids, second.token_ids)
            self.assertNotEqual(first.input_identity, second.input_identity)
            np.testing.assert_array_equal(np.asarray(first.input_embeddings)[7], np.ones(4))
