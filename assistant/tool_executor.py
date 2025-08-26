import datetime
import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Execute matched tools and return results.

    Adds logging at entry, parameter extraction, and on errors for easier debugging.
    """

    def __init__(self):
        self.tool_functions = {
            'get_time': self.get_time,
            'get_weather': self.get_weather,
            'turn_on_lights': self.turn_on_lights,
            'turn_off_lights': self.turn_off_lights,
            'play_music': self.play_music,
            'set_timer': self.set_timer
        }

        logger.debug(f"ToolExecutor initialized with tools: {list(self.tool_functions.keys())}")

    def execute_tool(self, tool_match: Dict, user_input: str) -> str:
        """Execute the matched tool and return a human-friendly result string."""
        logger.debug("execute_tool called", extra={"tool_match": tool_match, "user_input_preview": (user_input or '')[:160]})

        tool = tool_match.get('tool') if isinstance(tool_match, dict) else None
        if tool is None:
            logger.warning("execute_tool: no tool information provided")
            return "No tool information provided"

        tool_name = tool.get('tool_name')
        logger.info(f"Attempting to execute tool: {tool_name}")

        if tool_name in self.tool_functions:
            try:
                # Extract parameters if needed
                params = self.extract_parameters(tool, user_input)
                logger.debug(f"Extracted params for {tool_name}: {params}")
                result = self.tool_functions[tool_name](**params)
                logger.info(f"Tool {tool_name} executed successfully")
                logger.debug(f"Tool result: {result}")
                return result
            except Exception:
                logger.exception(f"Tool execution error for {tool_name}")
                return f"Sorry, I encountered an error executing {tool_name}"
        else:
            logger.warning(f"Tool {tool_name} requested but not implemented")
            return f"Tool {tool_name} is not implemented yet"

    def extract_parameters(self, tool: Dict, user_input: str) -> Dict:
        params = {}

        if not tool.get('input_required', False):
            logger.debug("No input required for this tool")
            return params

        user_input_lower = (user_input or '').lower()

        for var in tool.get('tool_input_variables', []):
            if var == 'location':
                location_match = re.search(r'in ([\w ]+)', user_input_lower)
                params['location'] = location_match.group(1).strip() if location_match else 'current location'

            elif var == 'room':
                rooms = ['bedroom', 'living room', 'kitchen', 'bathroom', 'all']
                for room in rooms:
                    if room in user_input_lower:
                        params['room'] = room
                        break
                else:
                    params['room'] = 'all'

            elif var == 'song_name':
                song_match = re.search(r'play (.+)', user_input_lower)
                params['song_name'] = song_match.group(1).strip() if song_match else 'random music'

            elif var == 'duration':
                duration_match = re.search(r'(\d+)\s*(minute|second|hour)', user_input_lower)
                if duration_match:
                    params['duration'] = f"{duration_match.group(1)} {duration_match.group(2)}s"
                else:
                    params['duration'] = '5 minutes'

        logger.debug(f"Parameters final: {params}")
        return params

    # Tool implementations
    def get_time(self) -> str:
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        return f"The current time is {current_time}"

    def get_weather(self, location: str = "current location") -> str:
        return f"The weather in {location} is sunny with 22 degrees Celsius"

    def turn_on_lights(self, room: str = "all") -> str:
        if room == "all":
            return "Turning on all lights"
        else:
            return f"Turning on {room} lights"

    def turn_off_lights(self, room: str = "all") -> str:
        if room == "all":
            return "Turning off all lights"
        else:
            return f"Turning off {room} lights"

    def play_music(self, song_name: str = "random music") -> str:
        return f"Now playing {song_name}"

    def set_timer(self, duration: str = "5 minutes") -> str:
        return f"Timer set for {duration}"
