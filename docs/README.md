# Helion Documentation

This directory contains the Sphinx documentation for the Helion project.

## Quick Start

### Prerequisites

Install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

### Building the Documentation

#### One-time Build

Generate static HTML documentation:

```bash
cd docs/
make html
```

The generated documentation will be available in `../site/` directory. Open `../site/index.html` in your browser to view the docs.

#### Development Mode with Live Reload

For active documentation development, use the live reload server:

```bash
cd docs/
make livehtml
```

This will:
- Build the documentation
- Start a local web server (typically at http://127.0.0.1:8000)
- **Automatically open your browser** to the documentation
- **Watch for file changes** and rebuild automatically
- **Refresh your browser** when changes are detected

The live reload server watches:
- All `.md` and `.rst` files in the docs directory
- Python source files (for autodoc changes)
- Configuration changes in `conf.py`

**Tip**: Keep this running while you edit documentation - changes will appear instantly in your browser!

#### Manual Commands

You can also use sphinx-autobuild directly with custom options:

```bash
# Basic live reload
sphinx-autobuild docs/ site/

# With custom port
sphinx-autobuild docs/ site/ --port 8080

# Watch additional directories (e.g., for theme development)
sphinx-autobuild docs/ site/ --watch ../helion/

# Force full rebuild on each change (useful for theme development)
sphinx-autobuild -a docs/ site/
```

### Cleaning Build Files

Remove all generated files:

```bash
cd docs/
make clean
```

### Debugging Builds

For verbose output during builds:

```bash
# See detailed build information
make html SPHINXOPTS="-v"

# See even more detail
make html SPHINXOPTS="-vv"

# Check for broken links
make linkcheck
```

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [MyST Parser Guide](https://myst-parser.readthedocs.io/)
- [sphinx-autobuild Documentation](https://github.com/sphinx-doc/sphinx-autobuild)
- [Read the Docs Theme](https://sphinx-rtd-theme.readthedocs.io/)
