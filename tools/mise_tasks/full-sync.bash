#!/usr/bin/env bash
#MISE description="Run full local developer environment sync"

set -eu -o pipefail

[ -n "${CI:-}" ] && exit 0 || true

mise run sync
if [ -d packages/sie_ts_sdk ]; then
	(
		cd packages/sie_ts_sdk
		mise exec -- pnpm install --frozen-lockfile
	)
fi
