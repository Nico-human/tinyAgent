from .custom_state import TodoState, UsageTrackState


# -- type hint for IDE --
class SessionState(TodoState, UsageTrackState):
    pass