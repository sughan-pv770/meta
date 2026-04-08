import uuid
from openenv.core.env_server import Environment
from models import YourAction, YourObservation, YourState

class YourEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self._state = YourState()

    def reset(self) -> YourObservation:
        self._state = YourState(episode_id=str(uuid.uuid4()))
        # Must return this specific result as per instructions
        return YourObservation(result="Ready", success=True)

    def step(self, action: YourAction) -> YourObservation:
        self._state.step_count += 1
        return YourObservation(result="Done", success=True)

    @property
    def state(self) -> YourState:
        return self._state
