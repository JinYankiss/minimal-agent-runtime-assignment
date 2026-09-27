class AgentError(Exception):
    """Base error for expected agent failures."""


class LLMProtocolError(AgentError):
    pass


class ToolNotFoundError(AgentError):
    pass


class ToolValidationError(AgentError):
    pass


class ToolExecutionError(AgentError):
    pass


class RetryNotSafeError(ToolExecutionError):
    pass


class ToolTimeoutError(ToolExecutionError):
    pass


class MaxRoundsExceeded(AgentError):
    pass
