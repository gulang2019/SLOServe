# Build each TeX target from its own directory so relative figure and
# bibliography paths keep working even when latexmk is launched from the
# repository root.
$do_cd = 1;
