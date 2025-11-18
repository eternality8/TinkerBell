import asyncio
import json
from tinkerbell.ai.orchestration import AIController
from tinkerbell.ai.client import AIStreamEvent
from types import SimpleNamespace

class StubClient:
    def __init__(self, events):
        self._batches = [list(events)]
        self.calls = []
        self.settings = SimpleNamespace(debug_logging=False)

    async def stream_chat(self, **kwargs):
        self.calls.append(kwargs)
        for event in self._batches[0]:
            yield event

sample_snapshot = {'selection': (0, 5), 'text': 'hello'}
tool_args = {'target_range': [0, 0], 'replacement_text': '# Title\n\nHello'}
sandwich = (
    'text'
    + '<|toolcallsbegin|><|toolcallbegin|>document_apply_patch<|toolsep|>'
    + json.dumps(tool_args, ensure_ascii=False)
    + '<|toolcallend|><|toolcallsend|>'
)
first_turn = [AIStreamEvent(type='content.delta', content=sandwich)]
second_turn = [AIStreamEvent(type='content.done', content='done')]
stub = StubClient(first_turn + second_turn)
controller = AIController(client=stub)
calls = []

class PatchTool:
    def run(self, **kwargs):
        calls.append(kwargs)
        return {'status': 'ok'}

controller.register_tool('document_apply_patch', PatchTool())

async def run():
    return await controller.run_chat('prompt', sample_snapshot)

result = asyncio.run(run())
print('tool_calls', result['tool_calls'])
print('calls', calls)
print('response', result['response'])
