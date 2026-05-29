from __future__ import annotations

import warnings


def pytest_configure() -> None:
    # TODO(tcombes): remove this once Pallas RNG generation avoids int64.
    # JAX x64 is disabled on TPU, so RNG-generated int64s are truncated and
    # spam Pallas test logs with one warning per generated statement.
    warnings.filterwarnings(
        "ignore",
        message=(
            "Explicitly requested dtype int64 requested in .* is not available, "
            "and will be truncated to dtype int32.*"
        ),
        category=UserWarning,
    )
