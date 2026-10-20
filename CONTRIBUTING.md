# Contributing to VocalMood AI

Thank you for your interest in contributing to **VocalMood AI**! We welcome improvements, bug reports, dataset calibration studies, and feature suggestions.

---

## Code of Conduct
Please ensure all communications remain respectful, collaborative, and inclusive.

---

## Development Workflow

1. **Fork the Repository:**
   Click the **Fork** button on GitHub to create your own branch copy.

2. **Clone your Fork:**
   ```bash
   git clone https://github.com/<your-username>/day-14-speech-emotion-voice-journal.git
   cd day-14-speech-emotion-voice-journal
   ```

3. **Create a Feature Branch:**
   ```bash
   git checkout -b feature/audio-filter-enhancement
   ```

4. **Set Up Development Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

5. **Run the Test Suite:**
   Before making changes, verify that all tests pass:
   ```bash
   pytest tests -v
   ```

6. **Implement Your Changes:**
   - Adhere to PEP 8 Python coding conventions.
   - Include docstrings for new functions and modules.
   - Add unit tests under `tests/` for any new logic or bug fixes.

7. **Verify Tests & Commit:**
   ```bash
   pytest tests -v
   git commit -m "feat(audio): add pre-emphasis filtering options"
   ```

8. **Submit a Pull Request:**
   Push your branch and open a PR with a description of what you changed, why it was needed, and any test results.
