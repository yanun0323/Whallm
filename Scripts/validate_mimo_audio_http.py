"""Full-checkpoint audio ingress smoke over authenticated Chat/Responses HTTP.

Supply two 24 kHz PCM16 WAVs speaking 'The secret number is forty two.' and
'The secret number is seventy three.' respectively. No external ASR is used.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import threading
from urllib.request import Request, urlopen

from deepseek_v4_ssd.generation import ModelRuntime
from deepseek_v4_ssd.manifest import InstalledModel
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_manager import ModelManager, ModelSpec, ModelDefaults
from deepseek_v4_ssd.server import OpenAIServer
from deepseek_v4_ssd.mimo.install import atomic_json, file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--wav42', type=Path, required=True)
    parser.add_argument('--wav73', type=Path, required=True)
    args = parser.parse_args()
    fixture_directory = args.report.parent / (args.report.stem + '-fixtures')
    if args.report.exists() or fixture_directory.exists():
        raise ValueError('Choose a new report path to preserve previous evidence.')
    fixture_directory.mkdir(parents=True)
    config = RuntimeConfig(slots=512, memory_limit_gib=28, prefill_step_size=32,
                           fp8_kv_cache=False, layer_major_prefill=False, prompt_cache_entries=2)
    loaded, tokens = [], []
    def load(spec):
        runtime = ModelRuntime(InstalledModel.open(args.model), config)
        original = runtime.stream
        def capture(*a, **kw):
            for piece in original(*a, **kw):
                tokens.append(piece.token)
                yield piece
        runtime.stream = capture
        loaded.append(runtime)
        return runtime
    manager = ModelManager([ModelSpec(id='mimo-v2.6-flash-rl', alias=None, path=str(args.model),
        model_kind='mimo-v2.6-flash-rl', runtime=config, defaults=ModelDefaults(12, 0, 1, 0))], runtime_loader=load)
    server = OpenAIServer(('127.0.0.1', 0), manager, api_key='local-audio-fixture', log_level='error')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    def request(path, data=None, mime='application/json', method=None):
        if isinstance(data, dict): data = json.dumps(data).encode()
        with urlopen(Request(base + path, data=data, method=method,
                headers={'Authorization':'Bearer local-audio-fixture','Content-Type':mime}), timeout=600) as response:
            return json.load(response)
    report = {'scope':'synthetic spoken-number HTTP ingress; not general audio quality, CUDA parity or App acceptance',
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'platform':platform.platform(),'python':platform.python_version(), 'config':asdict(config),
        'manifestSHA256':file_sha(args.model / 'manifest.json'),
        'cacheState':'new process; two Prompt Cache entries available, media bypass required; OS cache uncontrolled', 'results':[]}
    try:
        for source, expected, endpoint, key in ((args.wav42, '42', '/v1/chat/completions', 'messages'),
                                                (args.wav73, '73', '/v1/responses', 'input')):
            data = source.read_bytes()
            filename = 'number' + expected + '.wav'
            (fixture_directory / filename).write_bytes(data)
            asset = request('/api/assets', data, 'audio/wav')
            tokens.clear()
            result = request(endpoint, {'model':'mimo-v2.6-flash-rl', 'temperature':0,'seed':0,
                'max_tokens':12, key:[{'role':'user','content':[
                    {'type':'text','text':'What is the secret number spoken in this audio? Reply with only the digits.'},
                    {'type':'input_audio','file_id':asset['id']}]}]})
            if key == 'messages':
                answer = result['choices'][0]['message']['content']
            else:
                answer = ''.join(part.get('text','') for item in result['output'] for part in item.get('content',[]))
            print(filename, repr(answer), flush=True)
            report['results'].append({'fixture':str((fixture_directory / filename).relative_to(args.report.parent)), 'sha256':hashlib.sha256(data).hexdigest(),
                'endpoint':endpoint,'answer':answer,'expected':expected,'tokens':list(tokens),
                'outputTokenHash':hashlib.sha256(json.dumps(tokens).encode()).hexdigest()})
            if answer.strip() != expected:
                atomic_json(args.report, report)
                raise ValueError('Audio content was not correctly used')
            request('/api/assets/' + asset['id'], method='DELETE')
        report['promptCachesStored'] = len(loaded[0]._prompt_caches)
        if report['promptCachesStored']:
            raise ValueError('Audio must not acquire or store text Prompt Caches')
        atomic_json(args.report, report)
    finally:
        server.shutdown(); server.server_close(); thread.join(); manager.close()


if __name__ == '__main__':
    main()
