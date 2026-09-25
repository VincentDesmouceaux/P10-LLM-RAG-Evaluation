import logfire


_configured = False


def configure_observability() -> None:
    global _configured

    if _configured:
        return

    logfire.configure(
        send_to_logfire=False,
    )

    logfire.instrument_pydantic_ai()

    _configured = True
