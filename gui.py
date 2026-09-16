"""Luceo entry point. Run:  python gui.py"""

if __name__ == "__main__":
    # Imported here, not at module level: on Windows every CaImAn worker process
    # re-runs this file, and a top-level import made each one load the whole GUI
    # (~0.7 GB per worker before touching any data).
    from gui.app import main
    main()
