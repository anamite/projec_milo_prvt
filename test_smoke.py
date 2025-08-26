import json
from assistant.tool_matcher import ToolMatcher
from assistant.tool_executor import ToolExecutor

print("Running smoke tests...")
with open('config.json', 'r', encoding='utf-8') as f:
    cfg = json.load(f)

matcher = ToolMatcher(cfg.get('tools', []))
match = matcher.find_best_match('what time is it?')
print('Matcher result:', match)

executor = ToolExecutor()
# Test executing get_time tool
get_time_tool = next((t for t in cfg['tools'] if t['tool_name']=='get_time'), None)
if get_time_tool:
    res = executor.execute_tool({'tool': get_time_tool}, 'what time is it?')
    print('Executor get_time:', res)

print('SMOKE_OK')
