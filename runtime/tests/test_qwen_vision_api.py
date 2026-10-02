"""HTTP routing and early validation; FakeRuntime does not run the full model."""
import threading
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deepseek_v4_ssd.generation import ModelRuntime
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_manager import ModelManager, ModelSpec, ModelDefaults
from deepseek_v4_ssd.model_support import get_support
from deepseek_v4_ssd.server import OpenAIServer
from deepseek_v4_ssd.qwen_vision.processor import ImageInput
from runtime.tests import test_media_api
from runtime.tests.test_server import FakeRuntime
from runtime.tests.test_media import png


class QwenVisionAPITests(unittest.TestCase):
    request = test_media_api.MediaAPITests.request

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.runtimes = {}
        kinds = ('qwen3.8-flash-next', 'swift1.5-qwen3.8-flash-next')
        supports = [get_support(k) for k in kinds]
        cls.specs = [ModelSpec(id=s.descriptor.api_model_id, alias=f'vision-{i}',
            path=cls.temp.name, model_kind=s.descriptor.kind, runtime=RuntimeConfig(),
            defaults=ModelDefaults(24,0,1,0)) for i,s in enumerate(supports)]
        def load(spec):
            runtime=FakeRuntime()
            runtime.support=get_support(spec.model_kind)
            descriptor=runtime.support.descriptor
            runtime.installed=SimpleNamespace(root=Path(cls.temp.name),model_kind=spec.model_kind,
                model_id=descriptor.checkpoint_model_id,revision=descriptor.checkpoint_revision,
                maximum_context=262144,is_qwen=True,has_dspark=False)
            runtime.model=SimpleNamespace(mtp=None)
            runtime._codec=SimpleNamespace(encode=lambda *args:'prompt')
            def encode(messages,*args,**kwargs):
                runtime.last_encoded=ModelRuntime.encode_chat(runtime,messages,*args,**kwargs)
                return 'prompt'  # Only generation is faked; media validation is real.
            runtime.encode_chat=encode
            cls.runtimes[spec.id]=runtime
            return runtime
        cls.manager=ModelManager(cls.specs,runtime_loader=load,clear_cache=lambda:None)
        cls.server=OpenAIServer(('127.0.0.1',0),cls.manager,api_key='fixture-key',log_level='error')
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.base=f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join();cls.manager.close();cls.temp.cleanup()

    def test_both_endpoints_and_aliases_preserve_image_order(self):
        status,asset=self.request('/api/assets',data=png(),mime='image/png')
        self.assertEqual(status,201)
        try:
            with patch('deepseek_v4_ssd.qwen_vision.artifact.weight_path'):
                for spec in self.specs:
                    for endpoint,field in [('/v1/chat/completions','messages'),('/v1/responses','input')]:
                        status,result=self.request(endpoint,data={'model':spec.alias,field:[{'role':'user','content':[
                            {'type':'text','text':'before'},{'type':'input_image','file_id':asset['id']},
                            {'type':'text','text':'after'}]}]})
                        self.assertEqual(status,200,result)
                        content=self.runtimes[spec.id].last_encoded.messages[0]['content']
                        self.assertEqual(content[::2],('before','after'))
                        self.assertIsInstance(content[1],ImageInput)
        finally:
            self.request('/api/assets/'+asset['id'],method='DELETE')

    def test_missing_weights_mtp_and_invalid_image_fail_before_stream(self):
        _,asset=self.request('/api/assets',data=png(),mime='image/png')
        _,bad=self.request('/api/assets',data=b'not png',mime='image/png')
        try:
            for spec in self.specs:
                body={'model':spec.id,'stream':True,'messages':[{'role':'user','content':[
                    {'type':'input_image','file_id':asset['id']}]}]}
                status,result=self.request('/v1/chat/completions',data=body)
                self.assertEqual(status,400,result)
                self.assertIn('Verify and Repair',str(result))
                runtime=self.runtimes[spec.id];before=runtime.stream_call_count
                runtime.model.mtp=object()
                status,result=self.request('/v1/chat/completions',data=body)
                self.assertEqual(status,400,result);self.assertIn('MTP',str(result))
                runtime.model.mtp=None
                body['messages'][0]['content'][0]['file_id']=bad['id']
                with patch('deepseek_v4_ssd.qwen_vision.artifact.weight_path'):
                    status,result=self.request('/v1/chat/completions',data=body)
                    self.assertEqual(status,400,result)
                self.assertEqual(runtime.stream_call_count,before)
                # Missing supplemental image weights must not disable text inference.
                status,result=self.request('/v1/chat/completions',data={'model':spec.id,
                    'messages':[{'role':'user','content':'hello'}]})
                self.assertEqual(status,200,result)
        finally:
            for item in (asset,bad): self.request('/api/assets/'+item['id'],method='DELETE')
