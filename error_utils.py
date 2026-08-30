from __future__ import annotations


def iter_exception_chain(error: BaseException):
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def describe_exception(error: BaseException) -> str:
    message = str(error).strip()
    if message:
        return f"{type(error).__name__}: {message}"
    return type(error).__name__


def format_user_error(error: Exception, fallback: str) -> str:
    message = str(error).strip()
    if message:
        primary = message
    else:
        primary = f"{fallback} ({type(error).__name__})"

    for cause in list(iter_exception_chain(error))[1:]:
        cause_message = str(cause).strip()
        if not cause_message:
            continue
        if cause_message in primary:
            continue
        return f"{primary} Caused by: {cause_message}"

    return primary


def format_technical_error(error: Exception) -> str:
    return "\n".join(describe_exception(item) for item in iter_exception_chain(error))

