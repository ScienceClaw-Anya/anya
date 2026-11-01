"""Explicit provider routing, captured usage and no automatic request retries."""
from __future__ import annotations
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI
from .data import write_json

ALLOWED_NEMOTRON_HOSTS = {'integrate.api.nvidia.com', 'api.tokenfactory.us-central1.nebius.com'}

class ProviderUnavailable(RuntimeError):
    pass

class Provider:
    def __init__(self, project, name):
        self.project, self.name = Path(project), name
        load_dotenv(self.project / '.env', override=False)
        self.max_tokens = 32768
        self.reasoning = 'medium' if name == 'argon' else 'enabled'
        if name == 'argon':
            from neurotarget.scienceclaw_runtime import upstream_root
            upstream_root(self.project)
            from core.llm_client import LLMClient
            with contextlib.redirect_stdout(io.StringIO()):
                self.native = LLMClient(agent_name='anya-spes-v2-argon', backend='gemini')
            self.native.gemini_model = 'gemini-4-argon'
            self.native.gemini_reasoning_effort = 'medium'
            self.native.gemini_thinking_headroom = 0
            self.native.gemini_max_retries = 0
            self.native._coherence_shield_enabled = lambda: False
            self.native.gemini_client = self.native.gemini_client.with_options(max_retries=0, timeout=900)
            if urlparse(str(self.native.gemini_client.base_url)).hostname != 'generativelanguage.googleapis.com':
                raise ValueError('Gemini credentials require the official Google endpoint')
            self.model = 'gemini-4-argon'; self.base_url = str(self.native.gemini_client.base_url)
        elif name == 'nemotron':
            default_key = 'NEBIUS_TOKEN_FACTORY_API_KEY' if os.getenv('NEBIUS_TOKEN_FACTORY_API_KEY') else 'NVIDIA_API_KEY'
            key_env = os.getenv('V2_NEMOTRON_KEY_ENV', default_key)
            if key_env not in {'NEBIUS_TOKEN_FACTORY_API_KEY', 'NVIDIA_API_KEY'}:
                raise ValueError('Use a dedicated model API credential')
            key = os.getenv(key_env)
            if not key: raise ProviderUnavailable('Nemotron credential is not configured')
            expected_host = 'integrate.api.nvidia.com' if key_env == 'NVIDIA_API_KEY' else 'api.tokenfactory.us-central1.nebius.com'
            self.base_url = os.getenv('V2_NEMOTRON_BASE_URL', 'https://' + expected_host + '/v1/')
            parsed = urlparse(self.base_url)
            if parsed.scheme != 'https' or parsed.hostname != expected_host or parsed.query or parsed.fragment or parsed.username or parsed.password or parsed.port:
                raise ValueError('Unapproved Nemotron credential destination')
            self.model = os.getenv('V2_NEMOTRON_MODEL', 'nvidia/nemotron-3-super-120b-a12b')
            if self.model != 'nvidia/nemotron-3-super-120b-a12b':
                raise ValueError('This protocol requires the verified Nemotron 3 Super model')
            self.client = OpenAI(api_key=key, base_url=self.base_url, timeout=900, max_retries=0)
        else:
            raise ValueError('Unknown provider')

    def metadata(self):
        return {'provider': self.name, 'model': self.model, 'endpoint': self.base_url,
                'reasoning': self.reasoning, 'max_tokens': self.max_tokens, 'retries': 0}

    def ask(self, prompt, destination):
        destination = Path(destination)
        if destination.exists(): raise ValueError('Refusing to replay an already recorded API attempt')
        record = {**self.metadata(), 'status': 'attempt_reserved', 'captured_response': False,
                  'prompt_sha256': hashlib.sha256(json.dumps(prompt, separators=(',', ':')).encode()).hexdigest()}
        write_json(destination, record)
        started = time.perf_counter()
        response = None
        error = None
        try:
            if self.name == 'argon':
                resource = self.native.gemini_client.chat.completions
                original = resource.create
                def capture(*args, **kwargs):
                    nonlocal response, error
                    kwargs.pop('temperature', None)
                    try:
                        response = original(*args, **kwargs)
                        return response
                    except Exception as exc:
                        error = {'error_type': type(exc).__name__, 'http_status': getattr(exc, 'status_code', None)}
                        raise
                resource.create = capture
                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        text = self.native.call(json.dumps(prompt, separators=(',', ':')), max_tokens=self.max_tokens)
                finally:
                    resource.create = original
            else:
                response = self.client.chat.completions.create(model=self.model,
                    messages=[{'role': 'system', 'content': 'Use only supplied scientific evidence. Return the requested JSON object only.'},
                              {'role': 'user', 'content': json.dumps(prompt, separators=(',', ':'))}],
                    max_tokens=self.max_tokens, temperature=0, extra_body={'chat_template_kwargs': {'enable_thinking': True}})
                text = response.choices[0].message.content if response.choices else ''
        except Exception as exc:
            error = error or {'error_type': type(exc).__name__, 'http_status': getattr(exc, 'status_code', None)}
            text = None
        record['seconds'] = time.perf_counter() - started
        if response is not None:
            usage = getattr(response, 'usage', None)
            choice = response.choices[0] if response.choices else None
            record.update(status='response_captured', captured_response=True,
                response_model=getattr(response, 'model', None),
                usage=usage.model_dump() if usage is not None else None,
                finish_reason=getattr(choice, 'finish_reason', None),
                response_content=choice.message.content if choice and choice.message else None)
            write_json(destination, record)
            return text or ''
        record.update(status='uncaptured_failure', error=error, usage=None)
        write_json(destination, record)
        raise ProviderUnavailable('Provider request failed; see sanitized status, no automatic retry')

def parse_json(text):
    text = str(text).strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    return json.loads(text)

def preflight(project, output, names=('argon', 'nemotron'), attempt=1):
    result = {}
    for name in names:
        provider = Provider(project, name)
        folder = Path(output) / 'provider_preflight'; folder.mkdir(parents=True, exist_ok=True)
        destination = folder / f'{name}_{attempt:02d}.json'
        if destination.exists():
            saved = json.loads(destination.read_text())
            if saved.get('finish_reason') == 'stop' and parse_json(saved.get('response_content')) == {'ready': True} and saved.get('response_model') == provider.model:
                result[name] = {'cached_preflight': True, 'response_model': saved.get('response_model')}; continue
            raise ProviderUnavailable('A failed preflight exists; inspect it before any explicitly numbered new attempt')
        answer = provider.ask({'task': 'Interface check only. Return exactly {"ready":true}; no additional text.'}, destination)
        if parse_json(answer) != {'ready': True}: raise ValueError('Provider preflight returned the wrong JSON')
        saved = json.loads(destination.read_text())
        if saved.get('response_model') != provider.model or saved.get('finish_reason') != 'stop':
            raise ValueError('Preflight model identity or completion status differs from the requested protocol')
        result[name] = provider.metadata()
    write_json(Path(output) / 'provider_preflight/summary.json', result)
    return result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--providers', nargs='+', choices=['argon', 'nemotron'], default=['argon', 'nemotron'])
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(preflight(args.project, args.output, tuple(args.providers), args.attempt), indent=2))
