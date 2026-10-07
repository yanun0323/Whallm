from __future__ import annotations

import io
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import CacheList, KVCache
from PIL import Image

from deepseek_v4_ssd.media import ImagePart, MediaError, MediaRequest, ModalSpan, PreparedPrompt
from deepseek_v4_ssd.model_support import get_support
from deepseek_v4_ssd.model_support.catalog import BY_KIND
from deepseek_v4_ssd.qwen4_exp import ModelArgs, QSAAttention
from deepseek_v4_ssd.qwen_pooled_cache import QSAPooledIndexCache
from deepseek_v4_ssd.qwen_vision.artifact import catalog, definition, validate_runtime, weight_path
from deepseek_v4_ssd.qwen_vision.encoder import VisionEncoder, grid_coordinates, interpolation
from deepseek_v4_ssd.qwen_vision.inputs import ImagePrompt, generation_model, prepare
from deepseek_v4_ssd.qwen_vision.positions import MultimodalRoPE, image_positions
from deepseek_v4_ssd.qwen_vision.processor import ImageInput, image_input, validate_media
from deepseek_v4_ssd.tool_codec import ToolChoice

KINDS = ('qwen3.8-flash-next', 'swift1.5-qwen3.8-flash-next')


def image(color='red', size=(256, 256), format='PNG'):
    out = io.BytesIO()
    Image.new('RGB', size, color).save(out, format=format)
    return out.getvalue()


def request(*parts):
    return MediaRequest(({'role': 'user', 'content': tuple(parts)},), 'chat', (), ToolChoice(), 'low')


def installed(root, kind=KINDS[1]):
    d = BY_KIND[kind]
    return SimpleNamespace(root=Path(root), model_kind=kind, model_id=d.checkpoint_model_id,
        revision=d.checkpoint_revision, maximum_context=262144)


class QwenVisionTests(unittest.TestCase):
    def test_both_kinds_pin_separate_weight_sources_and_only_enable_images(self):
        data = catalog()
        self.assertEqual(len(data['tensors']), 333)
        for kind in KINDS:
            item = definition(installed('/tmp', kind))
            self.assertEqual(item['file']['size'], 897864704)
            self.assertTrue(BY_KIND[kind].supports('imageInput'))
            self.assertFalse(BY_KIND[kind].supports('audioInput'))
            self.assertFalse(BY_KIND[kind].supports('documentInput'))
        self.assertEqual(data['models'][1]['repository'], 'Yanun/Swift1.5-Qwen3.8-Flash-Next-Whallm-MXFP4')
        self.assertEqual(data['models'][0]['repository'], 'Qwen/Qwen3.8-Flash-Next-FP8')
        wrong = installed('/tmp')
        wrong.revision = '0'*40
        with self.assertRaises(MediaError): definition(wrong)

    def test_processor_normalization_patch_order_and_formats(self):
        for format, mime in [('PNG', 'image/png'), ('JPEG', 'image/jpeg'), ('WEBP', 'image/webp')]:
            part = image_input(ImagePart.from_bytes(image(format=format), mime))
            self.assertEqual(part.grid, (1, 16, 16))
            self.assertEqual(part.patches.shape, (256, 1536))
            self.assertEqual(part.patches.dtype, np.float32)
        part = image_input(ImagePart.from_bytes(image(), 'image/png'))
        patch = part.patches[0].reshape(3, 2, 16, 16)
        np.testing.assert_array_equal(patch[0], 1)
        np.testing.assert_array_equal(patch[1:], -1)
        pattern = np.arange(256*256*3, dtype=np.uint8).reshape(256,256,3)
        stream = io.BytesIO(); Image.fromarray(pattern).save(stream, format='PNG')
        patches = image_input(ImagePart.from_bytes(stream.getvalue(), 'image/png')).patches
        for index, (y,x) in enumerate(grid_coordinates((1,16,16))):
            expected = pattern[y*16:(y+1)*16, x*16:(x+1)*16].astype(np.float32)/255
            expected = (expected-.5)/.5
            expected = np.repeat(expected.transpose(2,0,1)[:,None], 2, axis=1).reshape(-1)
            np.testing.assert_allclose(patches[index], expected, rtol=0, atol=2e-7)

    def test_processor_bounds_and_animated_or_mislabeled_images(self):
        for data, mime in [(b'bad', 'image/png'), (image(), 'image/jpeg'),
                           (image(size=(1025,1024)), 'image/png'), (image(size=(1,512)), 'image/png')]:
            with self.subTest(mime=mime, size=len(data)), self.assertRaises(MediaError):
                image_input(ImagePart.from_bytes(data,mime))
        stream=io.BytesIO()
        Image.new('RGB',(32,32),'red').save(stream,format='PNG',save_all=True,
            append_images=[Image.new('RGB',(32,32),'blue')])
        with self.assertRaises(MediaError): image_input(ImagePart.from_bytes(stream.getvalue(),'image/png'))
        large = ImagePart.from_bytes(image(size=(1024,1024)), 'image/png')
        with self.assertRaisesRegex(MediaError, '2048'):
            validate_media(request(large, large, large))
        original = request('before', ImagePart.from_bytes(image(),'image/png'), 'after')
        validated = validate_media(original)
        self.assertIsInstance(original.messages[0]['content'][1], ImagePart)
        self.assertIsInstance(validated.messages[0]['content'][1], ImageInput)
        self.assertEqual(validated.messages[0]['content'][::2], ('before','after'))

    def test_spatial_positions_and_decode_delta_for_two_images(self):
        spans = [ModalSpan(2, 6, 'a', (1,4,6)), ModalSpan(10,4,'b',(1,4,4))]
        positions = image_positions(16, spans)
        np.testing.assert_array_equal(positions[:, :2], [[0,1]]*3)
        np.testing.assert_array_equal(positions[:,2:8],
            [[2]*6, [2,2,2,3,3,3], [2,3,4,2,3,4]])
        np.testing.assert_array_equal(positions[:,8:10], [[5,6]]*3)
        np.testing.assert_array_equal(positions[:,14:], [[9,10]]*3)
        rope = MultimodalRoPE(positions, ModelArgs())
        np.testing.assert_array_equal(np.array(rope.at(mx.array([0,15,16,17]))),
                                      [[0,10,11,12]]*3)
        for bad in [ModalSpan(2,5,'a',(1,4,6)), ModalSpan(0,4,'a',(2,2,4)), ModalSpan(15,4,'a',(1,4,4))]:
            with self.assertRaises(MediaError): image_positions(16,[bad])

    def test_interleaved_mrope_matches_independent_numpy_equations(self):
        args = ModelArgs()
        positions = image_positions(12, [ModalSpan(2,6,'a',(1,4,6))])
        rope = MultimodalRoPE(positions, args)
        values = np.arange(12*2*256,dtype=np.float32).reshape(1,12,2,256)/1000
        frequencies = 1/(args.rope_theta**(np.arange(0,64,2,dtype=np.float32)/64))
        angles = positions[:,:,None]*frequencies[None,None]
        mixed = angles[0].copy()
        mixed[:,1:33:3] = angles[1,:,1:33:3]
        mixed[:,2:30:3] = angles[2,:,2:30:3]
        angles = np.concatenate([mixed,mixed],axis=-1)[None,:,None].astype(np.float32)
        rotary=values[...,:64]
        expected = np.concatenate([rotary*np.cos(angles)+np.concatenate([-rotary[...,32:],rotary[...,:32]],axis=-1)*np.sin(angles),values[...,64:]],axis=-1)
        actual=rope.rotate(mx.array(values),mx.arange(12),1)
        np.testing.assert_allclose(np.array(actual),expected,rtol=1e-5,atol=2e-5)
        transposed=rope.rotate(mx.array(values.transpose(0,2,1,3)),mx.arange(12),2)
        np.testing.assert_array_equal(np.array(actual).transpose(0,2,1,3),np.array(transposed))

    def test_spatial_qsa_matches_chunked_and_pooled_cache_including_decode(self):
        mx.random.seed(2026)
        args=ModelArgs(hidden_size=16,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
            indexer_n_heads=2,indexer_head_dim=8,indexer_budget=4,indexer_compress_ratio=2,
            partial_rotary_factor=.5,rope_parameters={'mrope_section':[1,1,0]})
        layer=QSAAttention(args)
        rope=MultimodalRoPE(image_positions(14,[ModalSpan(3,4,'a',(1,4,4))]),args)
        x=mx.random.normal((1,19,16))
        expected=layer(x,CacheList(KVCache(),KVCache()),rope)
        for pooled in (False,True):
            cache=CacheList(KVCache(), QSAPooledIndexCache() if pooled else KVCache())
            chunks=[layer(x[:,a:b],cache,rope) for a,b in [(0,4),(4,9),(9,14),(14,15),(15,19)]]
            actual=mx.concatenate(chunks,axis=1)
            np.testing.assert_allclose(np.array(actual),np.array(expected),rtol=1e-4,atol=1e-5)
        text=layer(x,CacheList(KVCache(),KVCache()))
        self.assertGreater(np.max(np.abs(np.array(expected)-np.array(text))),1e-4)

    def test_tiny_encoder_matches_numpy_reference(self):
        mx.random.seed(42)
        config={'hidden_size':8,'in_channels':3,'temporal_patch_size':2,'patch_size':2,
            'spatial_merge_size':2,'num_heads':2,'intermediate_size':12,'out_hidden_size':6,
            'num_position_embeddings':9,'depth':2,'hidden_act':'gelu_pytorch_tanh'}
        model=VisionEncoder(config)
        model.patch_embed.proj.weight=mx.random.normal((8,3,2,2,2))*.1
        grid=(1,4,6)
        patches=mx.random.normal((24,24))
        actual=model(patches,grid)
        def array(x): return np.asarray(x.astype(mx.float32))
        def linear(x,l): return x@array(l.weight).T+array(l.bias)
        def norm(x,n):
            return (x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-6)*array(n.weight)+array(n.bias)
        def gelu(x): return x*.5*(1+np.vectorize(math.erf)(x/np.sqrt(2)))
        coords=grid_coordinates(grid)
        # Independent scalar bilinear interpolation, including patch merge order.
        table=array(model.pos_embed.weight).reshape(3,3,8)
        positional=[]
        for row,col in coords:
            y,x=row*2/3,col*2/5
            y0,x0=int(y),int(x); dy,dx=y-y0,x-x0
            positional.append(table[y0,x0]*(1-dy)*(1-dx)+table[y0,min(x0+1,2)]*(1-dy)*dx+
                table[min(y0+1,2),x0]*dy*(1-dx)+table[min(y0+1,2),min(x0+1,2)]*dy*dx)
        x=array(patches)@array(model.patch_embed.proj.weight).reshape(8,-1).T+array(model.patch_embed.proj.bias)+np.array(positional)
        angles=np.concatenate([coords,coords],axis=1)[:,None].astype(np.float32)
        for block in model.blocks:
            q,k,v=linear(norm(x,block.norm1),block.attn.qkv).reshape(24,3,2,4).transpose(1,0,2,3)
            def rotate(a): return a*np.cos(angles)+np.concatenate([-a[...,2:],a[...,:2]],axis=-1)*np.sin(angles)
            q,k=rotate(q).transpose(1,0,2),rotate(k).transpose(1,0,2)
            scores=q@k.transpose(0,2,1)/2
            weights=np.exp(scores-scores.max(-1,keepdims=True)); weights/=weights.sum(-1,keepdims=True)
            y=(weights@v.transpose(1,0,2)).transpose(1,0,2).reshape(24,8)
            x=x+linear(y,block.attn.proj)
            z=linear(norm(x,block.norm2),block.mlp.linear_fc1)
            z=z*.5*(1+np.tanh(np.sqrt(2/np.pi)*(z+.044715*z**3)))
            x=x+linear(z,block.mlp.linear_fc2)
        merger=model.merger
        expected=linear(gelu(linear(norm(x,merger.norm).reshape(-1,32),merger.linear_fc1)),merger.linear_fc2)
        np.testing.assert_allclose(np.array(actual),expected,rtol=2e-4,atol=2e-5)
        with self.assertRaises(ValueError): model(patches,(2,4,6))

    def test_missing_damaged_or_escaped_weights_and_mtp_fail_before_encoder(self):
        with tempfile.TemporaryDirectory() as root:
            install=installed(root)
            with self.assertRaisesRegex(MediaError,'Verify and Repair'): weight_path(install)
            path=Path(root)/'vision/common.bin'; path.parent.mkdir()
            path.write_bytes(b'bad')
            with self.assertRaises(MediaError): weight_path(install)
            runtime=SimpleNamespace(installed=install,model=SimpleNamespace(mtp=object()))
            with self.assertRaisesRegex(MediaError,'MTP'): validate_runtime(runtime)
            path.unlink(); path.symlink_to(Path(root)/'config.json')
            with self.assertRaises(MediaError): weight_path(install)

    def test_preparation_preserves_order_identity_and_releases_encoder_on_failure(self):
        runtime=SimpleNamespace(installed=installed('/unused'),
            model=SimpleNamespace(args=SimpleNamespace(hidden_size=4),
                model=SimpleNamespace(embed_tokens=lambda t: mx.zeros((len(t),4)))),
            tokenizer=SimpleNamespace(encode=lambda s,**kw:list(s.encode())),
            _codec=SimpleNamespace(encode=lambda messages,*args: ''.join(m['content'] for m in messages)))
        red=ImagePart.from_bytes(image(),'image/png'); blue=ImagePart.from_bytes(image('blue'),'image/png')
        encoder=lambda p,g:mx.ones((g[1]*g[2]//4,4))*p[0,0]
        with patch('deepseek_v4_ssd.qwen_vision.inputs.validate_runtime'), \
             patch('deepseek_v4_ssd.qwen_vision.inputs.load_encoder',return_value=encoder):
            first=prepare(runtime,validate_media(request('a',red,'b',blue,'c')))
            second=prepare(runtime,validate_media(request('a',blue,'b',red,'c')))
        self.assertEqual(first.token_ids,second.token_ids)
        self.assertNotEqual(first.input_identity,second.input_identity)
        self.assertEqual(first.spans[0].digest,red.digest)
        self.assertEqual(first.spans[1].digest,blue.digest)
        self.assertIsInstance(first,ImagePrompt)
        self.assertFalse(np.array_equal(np.array(first.input_embeddings),np.array(second.input_embeddings)))
        self.assertFalse(hasattr(runtime.model,'visual'))
        plain=PreparedPrompt(first.token_ids,first.input_embeddings,first.spans,first.input_identity)
        with self.assertRaises(MediaError): generation_model(runtime.model,plain)
        with patch('deepseek_v4_ssd.qwen_vision.inputs.validate_runtime'), \
             patch('deepseek_v4_ssd.qwen_vision.inputs.load_encoder',side_effect=RuntimeError('fixture')), \
             patch('deepseek_v4_ssd.qwen_vision.inputs.mx.synchronize') as drain:
            with self.assertRaises(RuntimeError): prepare(runtime,validate_media(request(red)))
            drain.assert_called_once()

    def test_text_requests_do_not_require_or_load_vision_weights(self):
        from deepseek_v4_ssd.generation import ModelRuntime
        for kind in KINDS:
            runtime=ModelRuntime.__new__(ModelRuntime)
            runtime.support=get_support(kind)
            runtime._codec=SimpleNamespace(encode=lambda *args:'unchanged text')
            with patch('deepseek_v4_ssd.qwen_vision.artifact.weight_path',side_effect=AssertionError('text touched vision')):
                self.assertEqual(runtime.encode_chat([{'role':'user','content':'hello'}]),'unchanged text')
