import httpx2
import openai
import pytest

from research_assistant.llm import IncompleteResponseError, is_retryable, with_retries

REQ = httpx2.Request("POST", "https://api.openai.com/v1/responses")


def timeout() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=REQ)


def status_error(cls: type[openai.APIStatusError], status: int, code: str | None = None):
    return cls("boom", response=httpx2.Response(status, request=REQ), body={"code": code})


class FakeFn:
    """Raises the given errors in order, then returns 'ok'."""

    def __init__(self, *errors: Exception):
        self.errors = list(errors)
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


# ---------- is_retryable (written for you; should pass right away) ----------


@pytest.mark.parametrize(
    "exc, expected",
    [
        (timeout(), True),
        (openai.APIConnectionError(request=REQ), True),
        (status_error(openai.RateLimitError, 429, "rate_limit_exceeded"), True),
        (status_error(openai.RateLimitError, 429, "insufficient_quota"), False),
        (status_error(openai.InternalServerError, 500), True),
        (status_error(openai.InternalServerError, 503), True),
        (status_error(openai.BadRequestError, 400), False),
        (status_error(openai.AuthenticationError, 401), False),
        (status_error(openai.NotFoundError, 404), False),
        (IncompleteResponseError("truncated"), False),
        (ValueError("our bug"), False),
    ],
)
def test_is_retryable(exc, expected):
    assert is_retryable(exc) is expected


# ---------- with_retries (YOU write it; these fail until you do) ----------


def test_returns_immediately_on_success():
    fn, sleeps = FakeFn(), []
    assert with_retries(fn, sleep=sleeps.append) == "ok"
    assert fn.calls == 1
    assert sleeps == []


def test_retries_transient_errors_then_succeeds():
    fn, sleeps, retries = FakeFn(timeout(), status_error(openai.InternalServerError, 503)), [], []

    result = with_retries(
        fn,
        max_attempts=3,
        sleep=sleeps.append,
        on_retry=lambda attempt, exc, delay: retries.append((attempt, type(exc).__name__)),
    )

    assert result == "ok"
    assert fn.calls == 3
    assert len(sleeps) == 2
    assert retries == [(2, "APITimeoutError"), (3, "InternalServerError")]


def test_never_retries_400():
    fn, sleeps = FakeFn(status_error(openai.BadRequestError, 400)), []

    with pytest.raises(openai.BadRequestError):
        with_retries(fn, sleep=sleeps.append)

    assert fn.calls == 1
    assert sleeps == []


def test_gives_up_after_max_attempts_and_raises_last_error():
    fn, sleeps = FakeFn(timeout(), timeout(), timeout(), timeout()), []

    with pytest.raises(openai.APITimeoutError):
        with_retries(fn, max_attempts=3, sleep=sleeps.append)

    assert fn.calls == 3  # 3 attempts total, not 3 retries
    assert len(sleeps) == 2  # no pointless sleep after the final failure


def test_backoff_grows_with_jitter_and_respects_cap():
    fn, sleeps = FakeFn(*[timeout() for _ in range(7)]), []  # fail every one of 7 attempts

    with pytest.raises(openai.APITimeoutError):
        with_retries(fn, max_attempts=7, base_delay=0.5, max_delay=4.0, sleep=sleeps.append)

    caps = [0.5, 1.0, 2.0, 4.0, 4.0, 4.0]
    assert len(sleeps) == len(caps)
    for delay, cap in zip(sleeps, caps, strict=True):
        assert cap / 2 <= delay <= cap, f"delay {delay} not in [{cap / 2}, {cap}]"
