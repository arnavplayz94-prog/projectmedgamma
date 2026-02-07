To restart the background server process:

1. **Stop the current process:**
   If running in a terminal: Press `Ctrl + C`.
   If running in background (this session): You can find the process ID and kill it, or simply close the terminal window where it started.

2. **Start the server:**
   Run the following command in your terminal:
   ```powershell
   python backend.py
   ```

**Note:**
- You do **not** need to set the API key manually anymore.
- I have created a `.env` file that `backend.py` will automatically load.
