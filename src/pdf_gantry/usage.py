"""Usage-error handling for the gantry Click group.

Click exits 2 on a usage error (unknown option, bad choice, missing
argument). Gantry already uses 2 for "ran fine, no results", so a typo
over a remote shell read as "paper not in library". ``GantryGroup`` maps
every ``click.UsageError`` to ``EXIT_USAGE`` (64, sysexits' EX_USAGE) and,
when ``--json`` appears anywhere in argv, reports it as a JSON object on
stdout instead of plain text on stderr.
"""

import json

import click

EXIT_USAGE = 64


class GantryUsageError(click.UsageError):
    """A usage error re-labelled with exit code 64 and optional JSON output."""

    exit_code = EXIT_USAGE

    def __init__(self, original: click.UsageError, as_json: bool):
        super().__init__(original.message, ctx=original.ctx)
        self.original = original
        self.as_json = as_json

    def usage_text(self) -> str:
        ctx = self.original.ctx
        return ctx.get_usage() if ctx is not None else ""

    def show(self, file=None) -> None:
        if not self.as_json:
            self.original.show(file)
            return
        click.echo(json.dumps({
            "error": self.original.format_message(),
            "usage": self.usage_text(),
            "exit_code": EXIT_USAGE,
        }))


def _json_requested(args) -> bool:
    return "--json" in list(args or [])


class GantryGroup(click.Group):
    """Top-level group: usage errors exit 64 and honour ``--json``.

    Usage errors surface in two places: parsing the group's own options
    (``make_context``) and parsing a subcommand's options (``invoke``, which
    recurses into nested groups). Both are wrapped; everything else --
    help, ``ctx.exit`` codes, standalone-mode printing -- is untouched.
    """

    def make_context(self, info_name, args, parent=None, **extra):
        as_json = _json_requested(args)
        try:
            ctx = super().make_context(info_name, args, parent=parent, **extra)
        except GantryUsageError:
            raise
        except click.UsageError as e:
            raise GantryUsageError(e, as_json) from e
        ctx.meta["gantry.json_argv"] = as_json
        return ctx

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except GantryUsageError:
            raise
        except click.UsageError as e:
            raise GantryUsageError(e, ctx.meta.get("gantry.json_argv", False)) from e
