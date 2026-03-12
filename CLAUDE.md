## Performance

When working with market/trading data, always prefer vectorized pandas/numpy operations over row-by-row loops. Never use iterrows() or apply() on large datasets without explicit approval.

## File Formatting

This is a Linux environment. All files must use Unix-style line endings (LF, `\n`). Never write files with Windows-style line endings (CRLF, `\r\n`). Shell scripts in particular will fail to execute if they contain `\r`. When writing shell scripts, always verify with `file <script>` after creation — if it reports "CRLF line terminators", fix with `sed -i 's/\r$//' <script>`.
