#!/bin/sh
# Open the app in a Chrome window without tabs or address bar, in English (presentation mode).
# A separate Chrome profile is used so the language and window flags always apply; your normal Chrome is untouched.
# Exit full screen with Cmd+Ctrl+F; close with Cmd+Q.
open -na "Google Chrome" --args --user-data-dir="$HOME/.heattents-present-chrome" --lang=en-US \
  --accept-lang=en-US,en --app=http://localhost:8502 --start-fullscreen --no-first-run --no-default-browser-check
