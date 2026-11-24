"""Turning numbers into human-friendly text for the user interface."""


def format_share(share: float) -> str:
    """Format a fraction as a percentage; show '<0.1%' instead of a misleading '0.0%'."""
    if 0 < share < 0.001:
        return "<0.1%"
    return f"{share:.1%}"


def format_duration(seconds: float) -> str:
    """Format a number of seconds as HH:MM:SS, e.g. 3899 -> '01:04:59'."""
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

