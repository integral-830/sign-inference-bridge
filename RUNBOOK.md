# ISL sign recognition on a Mac M2: complete step-by-step guide (final)

```
webcam -> MediaPipe (hands + body) -> neural net -> signs -> words -> sentences -> English (Groq)
                                                                         |
                                                      HTTP server on your Mac (port 8000)
                                                                         |  same Wi-Fi
                                                         phone browser / curl / ESP module
```

How to use this guide: go in order. Every step ends with a CHECKPOINT (what you should see). If a checkpoint fails,
use Part 13 (troubleshooting) before going on. Times and sizes are approximate.

| Part | What | Time |
|---|---|---|
| 1 | Install | 10 min |
| 2 | Check camera, MediaPipe models and the whole code | 10 min |
| 3 | Record your own signs (manual end-to-end) | 30-60 min |
| 4 | Dataset links and what each needs | reading |
| 5 | Letters (ISL alphabet + digits) | 30-60 min |
| 6 | Words (INCLUDE) | hours (downloads) |
| 7 | Optional: CISLR, ASL | optional |
| 8 | Clean up labels | 5 min |
| 9 | Train | 10-60 min |
| 10 | Run live (terminal sentences) | 10 min |
| 11 | ONE command: weights + HTTP server + live recognition | 10 min |
| 12 | ESP module and phone | 30 min |
| 13 | Test, improve, troubleshoot | ongoing |
| 14 | Command cheat sheet and file map | reference |

---

## PART 1. Install

```bash
cd ~/Downloads/sign-recognition-v2          # wherever you unzipped it
python3.11 -m venv .venv                    # Python 3.10-3.12 works; 3.11 is safest
source .venv/bin/activate                   # the prompt now starts with (.venv)
pip install --upgrade pip
pip install -r requirements.txt
```
- Every new terminal window: `cd` into the folder and run `source .venv/bin/activate` again.
- Use this fresh venv. Do not install other packages into it that change NumPy.
- Camera permission: System Settings > Privacy & Security > Camera > switch ON for Terminal (or your editor), then QUIT and reopen it.

CHECKPOINT: `pip` finishes without errors; `python -c "import numpy, cv2, torch; print(numpy.__version__, cv2.__version__, torch.__version__)"` prints three versions (NumPy 1.x).

---

## PART 2. Check camera, models and code

Run these in this order.

### 2.1 Download the MediaPipe models (one time)
```bash
python download_models.py
```
Downloads the hand and body-pose models into `mp_models/` with progress, a 30 s timeout, 3 retries and a size check.
A half-finished file is never kept. (Add `--lite` for the lighter pose model used by `--fast`, `--force` to start over.)
If you ever pressed Ctrl+C during an older download, first run `rm -rf mp_models`.
If your network blocks the download, use the `curl` commands from Part 13.

CHECKPOINT: `OK mp_models/hand_landmarker.task (... MB)`, `OK ... pose_landmarker_full.task`, `All models ready.`

### 2.2 Find the right camera
```bash
python camera_test.py --list
```
Probes cameras 0-4 and marks which ones give a real picture ("looks OK") and which are black.
On a Mac, camera 0 can be your iPhone (Continuity Camera). Use the number that looks OK as `--camera N` everywhere below.

### 2.3 See the live camera with hand and body drawing
```bash
python camera_test.py --camera 0 --mediapipe
```
A window shows the picture, fps, brightness and the hand/pose drawing. Click the window, press `q` to quit.
"BLACK IMAGE" means no permission, covered lens or wrong camera.

CHECKPOINT: you see yourself, your hands get dots and lines, and the text in the window shows about 15-30 fps.

### 2.4 Check the whole code
```bash
python check_setup.py
python selftest.py
```
CHECKPOINT: `check_setup.py` shows OK lines (packages, MPS GPU, MediaPipe backend, camera). `selftest.py` ends with
`11/11 checks passed`. MediaPipe log lines (I0000 / W0000) and a "[translate] request failed (offline)" line are normal.

---

## PART 3. Record your own signs (manual end-to-end)

Do this before any big download. It proves recording, training and live recognition work. Your own clips are also the
biggest accuracy boost later.

### 3.1 Set up the scene
- Camera at chest/face height, about an arm's length away. Head, shoulders and both hands visible.
- Light in front of you, not behind. Plain background if possible.
- Use the same setup when you later run live.

### 3.2 Choose labels
- One label per sign. Words: lowercase, no spaces (`hello`, `thank_you`). Letters/digits: ONE uppercase character (`A`, `B`, `1`).
- ALWAYS include `idle`: hands resting, lowered, or just not signing. It separates signs and ends sentences.
- First test: `idle,hello,thanks`.

### 3.3 Record
```bash
python collect_data.py --labels idle,hello,thanks --samples 30 --seconds 1.5 --auto
```
Add `--camera N` if camera 0 was not the right one.
- The window opens at once ("Starting... loading models"), then shows the camera. Click it, press SPACE to start.
  With `--auto` the next clips follow after a short countdown.
- Keys: SPACE start, `u` undo last clip, `n`/`p` next/previous label, `q` quit.
- Moving signs: do the whole sign inside the 1.5 s clip, then relax.
- Letters (held hand shapes): `--seconds 1.0`, hold still.
- `idle`: vary it (hands down, in lap, on a desk, scratching your face, looking away).
- Between clips change slightly: a bit closer or farther, small angle changes. Do not make every clip identical.
- Amounts: 30 per label for a test; 40-80 per label for real use.
- Other flags: `--no-pose` (hands only, for letters), `--seconds S`, `--samples N`.
- Clips where hands are not visible enough are discarded automatically ("discarded: hands not visible enough").

CHECKPOINT: `ls data/sequences` shows a folder per label; `ls data/sequences/hello | wc -l` is about 30. The window shows
`hands: 1` or `2` when it sees your hands, and a red warning if the picture is black.

### 3.4 Train a quick model
```bash
python train.py --model tcn --epochs 30
```
CHECKPOINT: prints a test accuracy and weakest classes, creates `model/sign_model.pt`. (GPU error? add `--device cpu`.)

### 3.5 Run it
```bash
python live_demo.py
```
Sign `hello`, drop your hands, wait about 2.5 s. CHECKPOINT: the terminal prints `[time] hello (9x%)` then `>> hello`.
`q` in the window quits. A transcript is saved in `transcripts/`.

### 3.6 Improve
1. Read the weakest classes printed by `train.py` (also `model/report.txt`).
2. Record 30-50 more clips of exactly those signs: `python collect_data.py --labels <weak ones> --samples 40 --auto`.
3. Train and test again. Repeat.

If your own vocabulary is all you need, skip to Part 11. Otherwise continue. Decide now: keep these clips (datasets get added
next to them) or start clean: `rm -rf data/sequences model`.

---

## PART 4. Dataset links and what each needs

| Dataset | Link | Size | Needs |
|---|---|---|---|
| ISL letters + digits (photos) | https://data.mendeley.com/datasets/7tsw22y96w | about 1 GB | unzip, `--auto-labels --no-pose` |
| ISL words, INCLUDE (263 signs, videos) | https://zenodo.org/records/4010759 | 56.8 GB total, take a few categories | unzip all parts into one folder, `--auto-labels` |
| INCLUDE project page | https://github.com/AI4Bharat/INCLUDE | info | |
| ISL words, CISLR (video + label file) | https://huggingface.co/datasets/Exploration-Lab/CISLR | check with `--dry-run` | free Hugging Face login, accept terms, `organize_videos.py` first |
| ASL letters (separate ASL project) | https://www.kaggle.com/datasets/grassknoted/asl-alphabet | about 1 GB | only the `asl_alphabet_train` folder |
| ISL sentence videos, TESTING only | https://data.mendeley.com/datasets/kcmpdxky7p and https://data.mendeley.com/datasets/98mzk82wbb | | sentence videos with text; use for `live_demo.py --video`, not for training |

Many datasets are research-use only. Check each license page before any commercial use.

Rule for `prepare_dataset.py`: the label is the folder that DIRECTLY contains the images/videos. It writes
`data/sequences/<label>/*.npy`. A dataset needs sorting only when it is NOT already one folder per sign.

| Dataset | Layout | Operation needed |
|---|---|---|
| Mendeley ISL letters | folder per sign (A, B, 1, 2, variants like E1) | none, `--auto-labels` merges variants |
| INCLUDE | category zips with one folder per sign (probably numbered like `1. Hello`; I could not confirm) | `setup_include.py` (list, then extract into one folder), then `--auto-labels` strips the numbers |
| CISLR | one videos folder + CSV/JSON | `organize_videos.py` sorts it, then `prepare_dataset.py` |
| ASL Alphabet | `asl_alphabet_train/<letter>/` | none (ignore the test folder) |

Look at any download before converting it (changes nothing):
```bash
python inspect_dataset.py <folder>
```
It lists the folders that hold media, file counts, what each label would be now and after `--auto-labels`, which folders
would merge, and folders with very few files.

---

## PART 5. Letters (ISL alphabet and digits)
1. Download https://data.mendeley.com/datasets/7tsw22y96w (Download all files), unzip to `~/Downloads/isl_alphabet`.
2. Inspect: `python inspect_dataset.py ~/Downloads/isl_alphabet`.
   CHECKPOINT: labels look like A, B, C ... 1, 2 ... (maybe E1, E2, 9a, 9b). If everything is one level deeper
   (`isl_alphabet/Something/A/...`), use that deeper folder as `--src`.
3. Convert (photos have no body, so `--no-pose`):
   ```bash
   python prepare_dataset.py --src ~/Downloads/isl_alphabet --max-per-class 200 --no-pose --auto-labels
   ```
   `--max-per-class 200` keeps a random 200 per sign. Interrupted? Run the same command again; it resumes.
   CHECKPOINT: `ls data/sequences` lists the letters/digits; the progress bar finished.

---

## PART 6. Words (INCLUDE)
1. Open https://zenodo.org/records/4010759. Each category is split into parts: download ALL parts of every category you take.
   Starter set, about 14 GB:

   | Category | Files | Size |
   |---|---|---|
   | Greetings | Greetings_1of2.zip, Greetings_2of2.zip | 1.6 GB, 1.2 GB |
   | Pronouns | Pronouns_1of2.zip, Pronouns_2of2.zip | 1.4 GB, 947 MB |
   | Days_and_Time | Days_and_Time_1of3.zip ... 3of3 | 1.2, 1.0, 0.85 GB |
   | People | People_1of5.zip ... 5of5 | about 1.3-1.6 GB each |

   More later: Home (4 parts), Places (4), Jobs (2), Society (3).
   Command line per file: `curl -L -C - -O "https://zenodo.org/records/4010759/files/<FILENAME>?download=1"` (`-C -` resumes an interrupted download).
2. Put all the zips you downloaded in one folder (for example `~/Downloads`; it only reads files ending in `.zip`).
   Keep the names Zenodo gave them (`Greetings_1of2.zip`, ...). They are separate zips that belong together: all parts of a
   category go into the SAME output folder, which the script does for you.
3. Look inside first, extracting nothing (fast, shows the real folder layout of YOUR download):
   ```bash
   python setup_include.py --zips ~/Downloads --only Greetings,Pronouns,Days_and_Time,People --list
   ```
   `--only` keeps the zips whose name starts with one of those words. Leave it out to take every zip in the folder.
   Check the table: per category the number of signs, videos, and videos per sign (min / median / max), the video types
   (.mp4, .mov), and the "label now -> label with --auto-labels" examples (for instance `1. Hello` -> `Hello`).
   CHECKPOINT: each sign has its own folder holding that sign's videos, and the labels look like words.
   If one folder mixes several signs, or the layout is different, paste the output to me and I will adapt the tools.
4. Extract everything into ONE folder:
   ```bash
   python setup_include.py --zips ~/Downloads --only Greetings,Pronouns,Days_and_Time,People --out ~/Downloads/include
   ```
   What it does for you:
   - checks every zip for corruption first and tells you which one to download again (`Home_1of2.zip: damaged or incomplete download`)
   - checks free disk space before starting (it needs about the unpacked size; override with `--force`)
   - merges all parts into the same folder, skips macOS junk (`__MACOSX`, `._files`, `.DS_Store`), refuses unsafe paths
   - is resumable: run the same command again after an interruption, finished files are skipped ("already there")
   - prints the same summary table for the unpacked folder
   - `--delete-zips` removes each zip after it extracted without problems (saves disk); `--no-verify` skips the corruption check
   Describe a folder any time: `python setup_include.py --summary ~/Downloads/include`.
   Plain `unzip -q` also works (`unzip -q Greetings_1of2.zip -d ~/Downloads/include`), but it does not check, merge-report or resume.
   If a zip says "not a valid zip", the download is incomplete (compare the size with the Zenodo page) or it is a split
   archive: download it again. A zip that asks for a password is not expected for INCLUDE: tell me.
5. Convert (videos contain a body, so no `--no-pose`). `--auto-labels` turns `1. Hello` into `Hello` and merges variants:
   ```bash
   python prepare_dataset.py --src ~/Downloads/include --max-per-class 80 --auto-labels --label-map labelmap_include.json
   ```
   `labelmap_include.json` renames the INCLUDE pronoun `I` to `me`. Why: a one-character label counts as a fingerspelled LETTER, so the pronoun would be mixed up with the letter I (and a `--subset letters` model would end up with only that one class).
   Already converted without it? Run `python merge_labels.py --map labelmap_include.json --apply`.
   Uses all CPU cores, resumable. A few unreadable videos are skipped with a message: normal.
   CHECKPOINT: `ls data/sequences` shows the word labels.

---

## PART 7. Optional extras
### 7.1 CISLR (more ISL words)
1. Open https://huggingface.co/datasets/Exploration-Lab/CISLR, log in, accept the terms.
2. Download:
   ```bash
   pip install -U huggingface_hub
   hf auth login                          # token from huggingface.co/settings/tokens
   hf download Exploration-Lab/CISLR --repo-type dataset --local-dir ~/Downloads/cislr --dry-run   # size first
   hf download Exploration-Lab/CISLR --repo-type dataset --local-dir ~/Downloads/cislr
   ```
   (If a `pip install` ever upsets NumPy, see Part 13.)
3. See what you got (names in `<...>` are placeholders):
   ```bash
   find ~/Downloads/cislr -maxdepth 2 | head -30
   python organize_videos.py --videos ~/Downloads/cislr/<videos-folder> --csv ~/Downloads/cislr/<labels-file>.csv --list-columns
   ```
   It prints the columns and its guess. If wrong: `--file-col NAME --label-col NAME`. JSON in WLASL format:
   `--wlasl-json file.json --top-classes 100 --splits train,val`.
4. Sort, inspect, convert:
   ```bash
   python organize_videos.py --videos ~/Downloads/cislr/<videos-folder> --csv ~/Downloads/cislr/<labels-file>.csv \
          --out ~/Downloads/cislr_sorted --min-per-class 2
   python inspect_dataset.py ~/Downloads/cislr_sorted
   python prepare_dataset.py --src ~/Downloads/cislr_sorted --max-per-class 40 --auto-labels
   ```
   Videos are symlinked (no extra disk); `--mode copy` copies. `--min-per-class` drops words with a single video.

### 7.2 ASL (a separate project)
```bash
python prepare_dataset.py --src ~/Downloads/asl/asl_alphabet_train --max-per-class 300 --no-pose
```
Use a separate copy of the project folder (or clear `data/sequences`) so ISL and ASL letters do not mix.

---

## PART 8. Clean up labels
The same sign under two names (`Hello`/`hello`, `E1`/`E2`, `hello_2`) must become one class.
```bash
ls data/sequences                                  # read the names
python merge_labels.py --auto --merge-case         # DRY RUN: prints what would be merged. READ IT.
python merge_labels.py --auto --merge-case --apply # do it (files are moved, never deleted)
```
- `--auto`: `E1`/`E2` -> `E`, `9a`/`9b` -> `9`, `hello_2` -> `hello`, `12. Hello` -> `Hello`. Plain numbers (`10`) stay.
- `--merge-case`: single characters UPPERCASE, everything else lowercase, so your `hello` and the dataset's `Hello` join.
  Use it only if upper/lower case never meant different signs. (Mixed labels like `Shunku` next to `idle` get lowercased: fine.)
- On a Mac, `Hello` and `hello` are the same folder; the tool handles the rename.
- Custom renames: copy `labelmap_example.json` to `mymap.json`, edit (e.g. `{"thank you": "thank_you"}`),
  run `python merge_labels.py --map mymap.json --apply`.

Check class sizes (small classes train badly):
```bash
for d in data/sequences/*/; do echo "$(ls "$d" | wc -l) $(basename "$d")"; done | sort -n | head -20
```
Classes under about 10 clips: record more (Part 3) or delete that folder.
Datasets have no "not signing" class, so ALWAYS also record `idle` yourself:
`python collect_data.py --labels idle --samples 80 --seconds 1.5 --auto`. Record your own versions of the words you use most.

---

## PART 9. Train
Two models (recommended when you use photo letters plus word videos):
```bash
python train.py --subset letters     # single characters + idle -> model/letters/sign_model.pt
python train.py --subset words       # everything else + idle   -> model/words/sign_model.pt
```
Why two: photo letters have no body, word videos do; in one model the net would learn "body visible = not a letter".

One model (only your own recordings, or only words):
```bash
python train.py                      # tries TCN, BiLSTM, Transformer, keeps the best -> model/sign_model.pt
```
Flags: `--model tcn` (fastest), `--epochs 80`, `--features hands|full|auto`, `--mirror-aug 0` (if direction matters), `--device cpu`.
CHECKPOINT: test accuracy printed, weakest classes and most-confused pairs listed, the `.pt` file(s) exist.
Dataset test accuracy is optimistic (same signers in train and test): the live demo is the real test.

---

## PART 10. Run live (terminal sentences)
```bash
python live_demo.py --weights model/words/sign_model.pt model/letters/sign_model.pt     # two models
python live_demo.py                                                                      # one model
python live_demo.py --video clip.mp4 --no-window                                         # transcribe a video file
```
- Window: current guess plus the sentence so far. Keys: `q` quit, `c` clear, BACKSPACE delete last word,
  ENTER end sentence now, `m` switch letters/words/auto.
- Terminal: each sign `[10:41:07] hello (93%)`, spelled words `spelled: HELLO`, finished sentences `>> GOOD HELLO`.
- A sentence ends after 2.5 s with hands out of view or in the `idle` pose. All sentences go to `transcripts/session-<time>.txt`.
- Spell with letters one after another (within 1.6 s). For double letters, relax or re-form the hand between them.

English sentences with Groq (optional):
```bash
pip install groq
export GROQ_API_KEY="gsk_..."                       # key from https://console.groq.com
python live_demo.py --weights model/words/sign_model.pt model/letters/sign_model.pt \
    --translate llm --translate-model openai/gpt-oss-20b
```
```
>> ME SCHOOL GO
   English: I am going to school.
```
- `openai/gpt-oss-20b` is fast (default); `openai/gpt-oss-120b` is slower and a bit better. Use the model names from your Groq console.
- What the model is told (context in `translate.py`): the signs come from a person who cannot speak (deaf / speech-impaired) and signs ISL; they were
  recognised automatically, so a sign may be wrong, repeated or missing; letters were already joined into words; the camera cannot see facial
  expression, so questions and negation must be inferred; how ISL grammar differs from English (time first, question word last, negation last,
  no articles); write ONE short first-person sentence for what the person most likely wants to say, without inventing details; examples included;
  if the signs make no sense it answers `(unclear) <the signs>`. The previous 3 sentences are sent only to resolve who/what is meant.
- Tell it about the signer (optional, helps it choose between readings; it may not invent details):
  `python run.py --translate-context "Ravi, a student; usually talks about school, family and food"`
  (also `--translate-context` on `live_demo.py`, or `SIGN_CONTEXT=...` in the environment or `.env`).
- To change the wording of the instructions, edit `SYSTEM` in `translate.py`.
- Only the sentence text is sent, never video. Without a key or internet it falls back to offline formatting (capitalisation only).
- Sign grammar differs from English: treat the English as a helpful reading, not an official interpretation.
- Extras: `--speak` (reads sentences aloud on macOS), `--wordlist words.txt` (snaps spelled words to your list), `--camera N`.

Tuning:
| Symptom | Try |
|---|---|
| Nothing printed | `--threshold 0.5`; the window should show hands; record more clips |
| Wrong signs appear | `--threshold 0.8` or `--min-votes 5` |
| Sentence ends too early/late | `--sentence-pause 3.5` / `1.8` |
| Low fps | `--fast` (run `python download_models.py --lite` once), `--stride 3` |
| Spelled letters form the wrong word | `--wordlist`, or lower `--letter-gap` |
| Letters model fires on moving words | raise `--motion-thr`, or press `m` |

---

## PART 11. ONE command: weights + HTTP server + live recognition

### 11.1 Quick start
```bash
cd ~/Downloads/sign-recognition-v2
source .venv/bin/activate
python run.py
```
(or double-click `start.command` in Finder; first time run `chmod +x start.command` in Terminal if macOS refuses.)

What `python run.py` does by itself:
1. Finds your weights: `model/words/sign_model.pt` + `model/letters/sign_model.pt`, else `model/sign_model.pt`, else any `sign_model.pt` under `model/`.
   No model found? It tells you to train first (Part 9).
2. Loads a Groq key from `GROQ_API_KEY` or from a `.env` file next to the script (a line `GROQ_API_KEY=gsk_...`; never share that file).
   English mode `auto`: Groq if a key and `pip install groq` exist, otherwise simple formatting.
3. Starts the HTTP server on port 8000 for all devices on your Wi-Fi (reuses one if it is already running there).
4. Starts the camera recognition and pushes (a) the sentence being signed right now and (b) every finished sentence.
5. Prints a banner with your Mac's Wi-Fi address and ready-to-paste curl commands:
```
 This Mac      : http://localhost:8000
 Other devices : http://192.168.1.50:8000      (phone / ESP / laptop on the SAME Wi-Fi)
   curl http://192.168.1.50:8000/live           # sentence being signed right now (words so far)
   curl http://192.168.1.50:8000/latest         # last FINISHED sentence
```
Quit with `q` in the camera window or Ctrl+C.

### 11.2 Options
| Command | Meaning |
|---|---|
| `python run.py --no-window` | terminal only, no camera window |
| `python run.py --camera 1` | another camera (see `camera_test.py --list`) |
| `python run.py --translate llm` | force Groq English (`off` = none, `simple` = formatting only) |
| `python run.py --translate-model openai/gpt-oss-120b` | different Groq model |
| `python run.py --translate-context "Ravi, a student; talks about school"` | facts about the signer for the English step (or `SIGN_CONTEXT` in `.env`) |
| `python run.py --token mysecret` | only senders with header `X-Token: mysecret` can POST (reading stays open) |
| `python run.py --port 9000` | another port |
| `python run.py --host 127.0.0.1` | this Mac only, no other devices |
| `python run.py --weights a.pt b.pt` | use these model files instead of auto-detect |
| `python run.py -- --threshold 0.7 --fast` | everything after `--` goes straight to `live_demo.py` |
| `python run.py --server-only` | just the server (no camera, no model): check that your phone reaches the Mac |
| `python run.py --simulate` | server + a fake signer that builds sentences word by word, to test phone/ESP with no camera and no model |

### 11.3 Test the whole chain WITHOUT signing (recommended first)
1. Terminal 1: `python run.py --simulate`
2. On the Mac: `curl http://localhost:8000/health` -> `ok`.
3. On a phone (same Wi-Fi): open `http://<Mac IP>:8000/health` -> `ok`; then reload `/live` (words appear one by one) and `/latest` (finished sentence).
CHECKPOINT: the phone shows `ok`, then growing text on `/live`, then `/latest` shows `I am going to school.`

### 11.4 Verify with real signing
Run `python run.py`, sign a few words:
- `curl http://<Mac IP>:8000/live` shows the words so far; stop signing for 2.5 s; `curl http://<Mac IP>:8000/latest` shows the finished sentence.
- Watch it change live from a terminal: `while true; do curl -s http://<Mac IP>:8000/live; sleep 0.5; done`

### 11.5 All server addresses (the same on Mac, phone, laptop, ESP)
| Request | What you get |
|---|---|
| `GET /health` | `ok` |
| `GET /live` | words of the sentence being signed right now (empty when idle), plain text |
| `GET /live.json` | `{"seq": 12, "text": "ME SCHOOL", "time": "..."}` |
| `GET /live/next?after=12&timeout=20` | waits until the live text changes (long poll); 204 on timeout |
| `GET /latest` | last FINISHED sentence as plain text (English if available, else glosses) |
| `GET /latest.json` | `{"id": 3, "gloss": "ME SCHOOL GO", "english": "I am going to school.", "text": "...", "time": "..."}` |
| `GET /next?after=3&timeout=20` | waits until a finished sentence newer than id 3 exists (long poll); 204 on timeout |
| `GET /history?n=20` | last 20 finished sentences as a JSON list (the server keeps 200) |
| `GET /esp?after=-1&timeout=20` | for ESP32 screens: plain `key=value` lines (`seq`, `live`, `id`, `text`, `gloss`, `time`), long poll; start with `after=-1`, then send back the `seq` you received; 204 when nothing changed |
| `POST /sentence` | `{"gloss": "...", "english": "..."}` stores a finished sentence (live_demo does this) |
| `POST /live` | `{"text": "..."}` sets the live text (live_demo does this) |

POST with a password: add header `X-Token: mysecret` (server started with `--token mysecret`).

How the transport works: HTTP long polling, not websockets. `/next` and `/live/next` hold the request open until something new
exists (answers within a fraction of a second) or answer 204 after the timeout; the client then asks again. Live text is sent
only when it changes, newest value wins if the server is slow, and it empties by itself when a sentence ends.
The push runs in a background thread, so a slow or missing server never slows the camera.

Manual alternative (server and camera in different terminals):
```bash
python server.py                                   # Terminal 1 (--port, --token also work)
python live_demo.py --weights model/words/sign_model.pt model/letters/sign_model.pt \
    --translate llm --push http://localhost:8000   # Terminal 2 (--push-token if the server has a token)
```

---

## PART 12. ESP module and phone

Mac and ESP must be on the SAME Wi-Fi. `localhost` works only on the Mac; devices use the Mac's Wi-Fi IP
(printed by `run.py`, or `ipconfig getifaddr en0`). If macOS asks "Allow python to accept incoming network connections?", click Allow.

### 12.1 Phone
Open `http://<Mac IP>:8000/live` and `http://<Mac IP>:8000/latest` in the browser (reload to refresh). If `/health` works there, the ESP will work.

### 12.2 ESP32 with a screen (recommended): `esp/sign_display/sign_display.ino`
Connects to your Wi-Fi by itself, shows and prints its IP address, asks the Mac for new text, and shows it on a 4-pin I2C OLED
with the SSD1306 driver (128x32 by default, 128x64 also supported):
- on every (re)connection (so also the first time): `WIFI OK` with the ESP32's IP address and the server address, held for 6 s,
  and the same printed to the Serial Monitor (SSID, ESP32 IP, gateway, signal, MAC, server). While idle the screen shows `READY` and the IP.
- while you sign: `SIGNING` and the last words (the end of the sentence if it is long)
- when a sentence finishes: `SENTENCE #n` and the sentence; big letters if it fits, small letters otherwise, and a long sentence is
  shown page by page (1.8 s per page) and stays on the last page
- `NO WIFI` / `NO SERVER` with a hint when something is wrong; signal strength (dBm) top right; the on-board LED blinks on a new sentence.
128x32 layout: one header row + 3 text rows of 21 characters (or 1 big row of 10). 128x64: 6 rows (or 3 big rows). Set `SCREEN_H` in the sketch (32 or 64).

What you need: an ESP32 board, the OLED, 4 jumper wires, a USB cable.
Wiring: OLED VCC -> 3V3, GND -> GND, SDA -> GPIO 21, SCL -> GPIO 22 (change `OLED_SDA`/`OLED_SCL` in the sketch for other pins).
The OLED's I2C address is found automatically (0x3C or 0x3D); the Serial Monitor lists every I2C device it sees at start-up.
1. Arduino IDE 2.x: File > Preferences > Additional boards manager URLs, add `https://espressif.github.io/arduino-esp32/package_esp32_index.json`;
   Tools > Board > Boards Manager > install "esp32 by Espressif Systems".
2. Tools > Manage Libraries: install "Adafruit SSD1306" and "Adafruit GFX Library" (accept the dependencies).
3. Open `esp/sign_display/sign_display.ino`. Under SETTINGS set `WIFI_SSID`, `WIFI_PASS` and `SERVER` (`http://<Mac IP>:8000`, the "Other devices" address printed by `run.py`); check `SCREEN_H`.
4. Tools > Board > "ESP32 Dev Module" (or your board), pick the Port, click Upload. If it hangs on "Connecting...", hold the BOOT button.
5. Tools > Serial Monitor at 115200 baud.
6. On the Mac: `python run.py --simulate` first (fake sentences, no camera needed).
CHECKPOINT 1: the screen shows `WIFI OK` + `IP 192.168.x.x` for 6 s and the Serial Monitor prints the "Wi-Fi connected" block with the ESP32 IP.
CHECKPOINT 2: then `SIGNING` with words appearing one by one, then `SENTENCE #1` with `I am going to school.`
Then stop the simulation and use `python run.py` with real signing.
Notes: the ESP32 supports 2.4 GHz Wi-Fi only (not 5 GHz). Blank screen: check SDA/SCL/VCC/GND, and what the Serial Monitor says about "I2C devices found" (none = wiring).
A different screen (TFT, TTGO T-Display, M5Stack): only `drawFrame()` and the display setup in the sketch change; tell me which screen and I will adapt it.
How it talks to the Mac: one long-poll request `GET /esp?after=<seq>&timeout=20` (the server answers as soon as the live text or the last sentence changes, otherwise 204 after 20 s). No JSON library is needed.

### 12.3 ESP32 / ESP8266 without a screen (Serial Monitor only, no libraries)
1. Install the ESP32 or ESP8266 board package and select your board.
2. Open `esp/sign_receiver/sign_receiver.ino`. Set `WIFI_SSID`, `WIFI_PASS`, and `SERVER` (`http://<Mac IP>:8000`).
3. Upload, open Serial Monitor at 115200 baud.
4. Sign a sentence on the Mac.

CHECKPOINT: `SENTENCE: I am going to school.` and `GLOSS   : ME SCHOOL GO` appear.

5. Put your own action in `onSentence()` (display, speech, LED, motor).

The sketch receives each FINISHED sentence through `/next` (long poll, answers almost instantly). Simplest alternatives on any
device: `GET /latest` every second and compare to the previous text, or `GET /live` to show the words as they are signed.
The sketch does not use `/live/next` yet; to show live words on the ESP, change the URL to `/live/next?after=<seq>` and read `"seq"`
instead of `"id"`. Tell me if you want me to add this.

### 12.4 ESP troubleshooting
| Problem | Fix |
|---|---|
| Screen says `NO SERVER (-1)` / Serial shows `request failed (-1)` | wrong IP, other Wi-Fi, or the Mac firewall blocked python: restart `run.py` and click Allow |
| Works on the Mac, not on the ESP/phone | you used `localhost`; use the Mac's Wi-Fi IP. Guest Wi-Fi / "client isolation" also blocks it |
| The Mac's IP changed | give the Mac a fixed IP in the router, or re-check `ipconfig getifaddr en0` |
| Nothing arrives | the `run.py` terminal must show `stored #N`; if not, no sentence finished yet (wait 2.5 s after signing) or nothing was recognised |
| Security | home network only; never forward the port to the internet; use `--token` if others share your Wi-Fi |

---

## PART 13. Test, improve, troubleshoot

Test loop:
1. Record 20-50 sentences (phone or webcam) using only words you trained and that are not in your training clips.
2. `python live_demo.py --video sentence.mp4 --no-window`; compare with what you signed. Count wrong / missing / extra words.
3. Record 30-50 more clips of the signs that failed (Part 3.3), retrain (Part 9), test again.
Sentence videos from ISL-CSLTR / ISLVT (Part 4) also work as test input; many of their words will not be in your vocabulary.

Troubleshooting:
| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | venv not active (`source .venv/bin/activate`) or `pip install -r requirements.txt` did not finish |
| Camera window black/empty, or the program exits at once | `python camera_test.py --list`, then `--camera N`. Causes: Terminal not allowed under Privacy > Camera (allow, then QUIT and reopen Terminal); camera 0 is your iPhone (turn off Continuity Camera in iPhone Settings > General > AirPlay & Continuity, or use `--camera 1`); another app (Zoom, FaceTime, a browser tab) uses the camera. The collector shows "Starting..." while models load and a red "BLACK IMAGE" warning if the picture is black. Click the window so it gets key presses |
| `Downloading MediaPipe model ...` hangs | `rm -rf mp_models` then `python download_models.py`. Blocked network: `mkdir -p mp_models && curl -L -o mp_models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task` and `curl -L -o mp_models/pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task` |
| `ImportError: numpy._core.multiarray failed to import` | NumPy and OpenCV do not match (a later `pip install` changed NumPy). `pip install --force-reinstall "numpy<2"` then `pip install --force-reinstall -r requirements.txt`; check `python -c "import numpy, cv2; print(numpy.__version__, cv2.__version__)"`. `check_setup.py` prints this hint |
| "model made by old v1 scripts" | old data/models are incompatible: redo `prepare_dataset.py` / `collect_data.py` and train |
| GPU error in training | `--device cpu` |
| Disk full | delete dataset zips and unzipped videos after conversion |
| Low accuracy | check `idle` exists; live distance/light must match your recordings; record your own clips of the same words; start with 20-50 words, not 200 words x 15 clips |
| Groq errors | check `GROQ_API_KEY` and that the model name is in your Groq console; the demo falls back to offline formatting and prints one warning |
| Every sign shows as `I (100%)` (or one sign at 100%) | the letters model has only ONE class (`model/letters ... 1 classes` in the start-up line): usually the INCLUDE pronoun `I` was the only single-character label. Fix: `python merge_labels.py --map labelmap_include.json --apply`, `rm -rf model/letters`, then either convert real letters (Part 5) and `python train.py --subset letters`, or run words only: `python live_demo.py --weights model/words/sign_model.pt`. `train.py` now refuses to train fewer than 2 classes and `live_demo.py` refuses to load such a model |
| The words model names a sign (e.g. `RACE ETHNICITY`) when you are not signing | there is no `idle` class in the data. Record it: `python collect_data.py --labels idle --samples 80 --seconds 1.5 --auto`, then `python train.py --subset words`. `train.py` and `live_demo.py` now warn when `idle` is missing. Also try `--threshold 0.8` |
| `run.py`: "No trained model found" | train first (Part 9) or test the server with `python run.py --simulate` |
| `run.py`: "Port 8000 is in use" | `python run.py --port 8001` (or stop the other program) |

---

## PART 14. Cheat sheet and file map

Daily flow after everything is trained:
```bash
cd ~/Downloads/sign-recognition-v2 && source .venv/bin/activate
python run.py                  # camera + server; use the printed address on the phone/ESP
```
Add more signs:
```bash
python collect_data.py --labels <signs> --samples 40 --auto
python train.py --subset words          # and/or --subset letters
```

| What | Where |
|---|---|
| Your recordings and converted datasets | `data/sequences/<label>/*.npy` |
| Trained models | `model/sign_model.pt`, or `model/letters/` and `model/words/` (plus `report.txt`) |
| Transcripts | `transcripts/session-<time>.txt` |
| MediaPipe models | `mp_models/` |
| Groq key (optional) | environment variable or `.env` |
| One-command launcher | `run.py`, `start.command` |
| Server / pusher / ESP sketches | `server.py`, `push.py`, `esp/sign_display/sign_display.ino` (screen), `esp/sign_receiver/sign_receiver.ino` (Serial only) |
| Camera and model checks | `camera_test.py`, `download_models.py`, `check_setup.py`, `selftest.py` |
| Data tools | `collect_data.py`, `setup_include.py`, `prepare_dataset.py`, `inspect_dataset.py`, `organize_videos.py`, `merge_labels.py` |
| Training / live | `train.py`, `model.py`, `live_demo.py`, `sentence.py`, `fusion.py`, `translate.py` |

Honest status: `selftest.py` passed 11/11 on your Mac (PyTorch on MPS, MediaPipe, training, sentence logic, label merging,
sorter, fusion, translation with a fake client). The server, push, long polling, `run.py`, the model downloader (including
stalls and Ctrl+C) and the collector's camera handling were tested here with simulated cameras and local servers. Not yet
verified on real hardware: your camera loop end to end, a real Groq call, INCLUDE's and CISLR's actual folder/column layouts,
and the ESP sketch on a board. Paste me any output that looks wrong and I will adjust.
