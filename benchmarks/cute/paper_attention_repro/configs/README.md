# Final GB300 attention configurations

- `forward/`: 87 passing forward configs; 11 reuse previous-session GB300 winners.
- `backward/`: 84 passing backward cases, each with preprocess, main, and combined JSON (252 files).
- `manifest.json`: original paths and SHA256s for all 339 files.

Every config was checked against its final PASS result. All implementations of
each case and variant used the same physical GPU. No paper B200 config is
selected in the final results. The three unavailable Helion MLA backward cases
retain SKIP status and have no winning config.
