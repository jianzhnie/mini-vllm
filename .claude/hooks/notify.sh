#!/bin/bash
# Notification hook: WeChat + desktop + voice when Claude needs attention
set -euo pipefail

INPUT=$(cat)
NOTIFY_TYPE=$(jq -r '.notification_type // "task"' <<<"$INPUT")
MESSAGE=$(jq -r '.message // ""' <<<"$INPUT")

# Build a short spoken summary
case "$NOTIFY_TYPE" in
    permission)
        VOICE_TEXT="需要授权"
        ;;
    *)
        VOICE_TEXT="任务完成，请确认"
        ;;
esac

# WeChat notification via Server酱
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/Users/robin/work_dir/mini-vllm}"
if [[ -f "$PROJECT_DIR/.claude/.env" ]]; then
    SCT_SENDKEY=$(grep '^SCT_SENDKEY=' "$PROJECT_DIR/.claude/.env" | cut -d= -f2-)
    if [[ -n "${SCT_SENDKEY:-}" ]]; then
        (
            curl -s -X POST "https://sctapi.ftqq.com/${SCT_SENDKEY}.send" \
                -d "title=${VOICE_TEXT}" \
                -d "desp=${MESSAGE:-Claude 需要你的关注}" \
                -o /dev/null
        ) &
    fi
fi

# Desktop notification (macOS)
osascript -e "display notification \"${MESSAGE:-Claude 需要你的关注}\" with title \"Claude Code\"" 2>/dev/null || true

# Voice notification — run in background to avoid blocking
(say "$VOICE_TEXT" 2>/dev/null; say "$VOICE_TEXT" 2>/dev/null) &

exit 0
