# Sign recognition v2: ISL (and ASL) on a Mac M2

Webcam -> MediaPipe (hands + body) -> neural net -> signs -> words -> sentences -> English (Groq)
-> HTTP server on your Mac -> phone / ESP32 screen.

START HERE: open RUNBOOK.md. It is the complete step-by-step guide (install, record your own signs, datasets,
training, live run, ESP32, troubleshooting).

Quick start (details in RUNBOOK.md):

    cd sign-recognition-v2
    python3.11 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python download_models.py        # one-time MediaPipe models
    python camera_test.py --list     # which camera works
    python selftest.py               # checks the install, expect 11/11
    python collect_data.py --labels idle,hello,thanks --samples 30 --auto
    python train.py --model tcn --epochs 30
    python run.py                    # camera + HTTP server + prints curl commands for phone/ESP
    python run.py --simulate         # test phone/ESP without camera or model

What is in this folder
    RUNBOOK.md               the guide (read this first)
    run.py, start.command    ONE command: weights + HTTP server + live recognition
    server.py, push.py       sentence server (long polling) and the sender used by live_demo.py
    esp/sign_display/        ESP32 + OLED screen sketch  (flash this one)
    esp/sign_receiver/       ESP32 Serial-Monitor-only sketch (no screen)
    collect_data.py          record your own signs
    setup_include.py         unzip + check the INCLUDE dataset zips
    prepare_dataset.py       convert photos/videos to training data
    inspect_dataset.py       look at a dataset before converting it
    organize_videos.py       sort CSV/JSON-labelled videos (CISLR ...) into folders
    merge_labels.py          merge duplicate labels (E1/E2 -> E, Hello/hello)
    labelmap_example.json, labelmap_include.json   label maps (include: pronoun I -> me)
    train.py, model.py       training (TCN / BiLSTM / Transformer on the Apple GPU)
    live_demo.py             live recognition with sentences (+ translate.py, sentence.py, fusion.py)
    camera_test.py, download_models.py, check_setup.py, selftest.py   checks and one-time setup
    common.py, labels.py     shared code
    requirements.txt

Folders that appear while you work: data/sequences (your training data), model/ (trained weights),
mp_models/ (MediaPipe models), transcripts/ (saved sentences). Nothing else is needed.
