/*
  ESP32 sign-language display  (SSD1306 OLED, 128x32 by default; 128x64 also works)
  ---------------------------------------------------------------------------------
  Connects to your Wi-Fi, SHOWS AND PRINTS ITS IP ADDRESS, then asks the sentence server on your Mac (python run.py)
  for new text and shows it on the OLED:
     - while you sign: the words so far ("SIGNING")
     - when a sentence is finished: the sentence ("SENTENCE #n"); long sentences are shown page by page

  HOW TO FLASH (Arduino IDE 2.x)
   1. File > Preferences > "Additional boards manager URLs": add
        https://espressif.github.io/arduino-esp32/package_esp32_index.json
      Tools > Board > Boards Manager > install "esp32 by Espressif Systems".
   2. Tools > Manage Libraries: install "Adafruit SSD1306" and "Adafruit GFX Library" (accept the dependencies).
   3. Tools > Board > "ESP32 Dev Module" (or your exact board), pick the Port.
   4. Edit the 3 lines under SETTINGS (Wi-Fi name, password, Mac address). Click Upload.
      (If upload hangs on "Connecting...", hold the BOOT button until it starts.)
   5. Tools > Serial Monitor, 115200 baud.

  WIRING (4-pin I2C OLED, SSD1306)
     OLED VCC -> ESP32 3V3      OLED GND -> GND
     OLED SDA -> GPIO 21        OLED SCL -> GPIO 22        (change OLED_SDA / OLED_SCL below for other pins)
  The I2C address is found automatically (0x3C or 0x3D); the Serial Monitor lists every I2C device it sees.
  Screen size: SCREEN_H 32 (128x32) or 64 (128x64) below. Without a screen the sketch still works and prints to Serial.

  WHAT YOU SEE
   - On every (re)connection: "WIFI OK" with the ESP32's IP address and the server address, held for IP_SHOW_MS
     (6 s). The same is printed to the Serial Monitor. While idle the screen also shows the IP.
   - NO WIFI / NO SERVER with a hint when something is wrong.

  OTHER SCREENS (TFT, TTGO T-Display, M5Stack): only drawFrame() and the display setup need to change.
  Needs the sentence server running on the Mac:  python run.py   (or  python run.py --simulate  for fake sentences).
  The ESP32 and the Mac must be on the same Wi-Fi (ESP32 supports 2.4 GHz only).
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <vector>

// ============================== SETTINGS ==============================
const char* WIFI_SSID = "YOUR_WIFI_NAME";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";
// IP of your Mac on the same Wi-Fi. run.py prints it as "Other devices : http://192.168.x.x:8000"
const char* SERVER    = "http://192.168.1.50:8000";

#ifndef SCREEN_H
#define SCREEN_H   32          // 32 = 128x32 screen, 64 = 128x64 screen
#endif
#define SCREEN_W   128
#define OLED_ADDR  0x3C        // tried first; 0x3D is tried automatically if nothing answers
#define OLED_SDA   21
#define OLED_SCL   22
#define LED_PIN    2           // on-board LED blinks on a new sentence (-1 = off)
#define IP_SHOW_MS 6000        // how long the IP screen stays after (re)connecting
#define PAGE_MS    1800        // time per page for sentences longer than the screen
// =======================================================================

Adafruit_SSD1306 display(SCREEN_W, SCREEN_H, &Wire, -1);
bool haveDisplay = false;

// Layout: one header row (8 px), then the body. On 128x32: 3 text rows; on 128x64: 6 text rows.
const int BODY_TOP = (SCREEN_H >= 64) ? 13 : 8;
const int BODY_H   = SCREEN_H - BODY_TOP;

struct State {
  long   seq = -1;     // changes whenever the live text or the last sentence changes
  String live;         // words of the sentence being signed right now ("" when idle)
  long   id = 0;       // number of the last finished sentence
  String text;         // last finished sentence (English if available, else the signs)
  String gloss;        // last finished sentence as signs
};

State  cur;            // what is on the screen now
long   afterSeq = -1;  // what we told the server we already have (-1 = send me the current state)
String statusLine = "starting";
int    lastCode = 0;

// ---------------------------------------------------------------- helpers (pure functions)

// Keep printable ASCII only (the default OLED font has nothing else).
String asciiOnly(const String& in) {
  String out;
  for (unsigned int i = 0; i < in.length(); i++) {
    char c = in[i];
    out += (c >= 32 && c <= 126) ? c : '?';
  }
  return out;
}

// Word-wrap `s` into lines of at most `cols` characters (long words are cut). Appends to `lines`.
void wrapText(const String& s, int cols, std::vector<String>& lines) {
  String line, word;
  for (unsigned int i = 0; i <= s.length(); i++) {
    char c = (i < s.length()) ? s[i] : ' ';
    if (c != ' ') { word += c; continue; }
    if (word.length() == 0) continue;
    while ((int)word.length() > cols) {            // word longer than a whole line
      if (line.length() > 0) { lines.push_back(line); line = ""; }
      lines.push_back(word.substring(0, cols));
      word = word.substring(cols);
    }
    if (line.length() == 0)                                    line = word;
    else if ((int)(line.length() + 1 + word.length()) <= cols) line += " " + word;
    else { lines.push_back(line); line = word; }
    word = "";
  }
  if (line.length() > 0) lines.push_back(line);
}

// Like wrapText, but honours explicit '\n' (each paragraph is wrapped separately).
void wrapParagraphs(const String& s, int cols, std::vector<String>& lines) {
  int from = 0;
  while (from <= (int)s.length()) {
    int nl = s.indexOf('\n', from);
    if (nl < 0) nl = s.length();
    wrapText(s.substring(from, nl), cols, lines);
    from = nl + 1;
  }
}

// SERVER without "http://", so it fits a 21-character row (192.168.1.50:8000).
String serverShort() {
  String s = String(SERVER);
  if (s.startsWith("http://")) s = s.substring(7);
  return s;
}

// Parse the plain-text answer of GET /esp  ("key=value" per line). Returns false if it has no seq.
bool parseState(const String& body, State& st) {
  bool gotSeq = false;
  int start = 0;
  while (start <= (int)body.length()) {
    int end = body.indexOf('\n', start);
    if (end < 0) end = body.length();
    String line = body.substring(start, end);
    line.trim();
    int eq = line.indexOf('=');
    if (eq > 0) {
      String key = line.substring(0, eq);
      String val = asciiOnly(line.substring(eq + 1));
      if      (key == "seq")   { st.seq = val.toInt(); gotSeq = true; }
      else if (key == "live")  st.live  = val;
      else if (key == "id")    st.id    = val.toInt();
      else if (key == "text")  st.text  = val;
      else if (key == "gloss") st.gloss = val;
    }
    start = end + 1;
  }
  return gotSeq;
}

// Choose text size and wrap. Big text (size 2: 10 characters per row) if the whole text fits, else size 1 (21 per row).
void layoutText(const String& body, bool allowBig, std::vector<String>& lines, int& size) {
  lines.clear();
  size = 1;
  if (allowBig) {
    wrapParagraphs(body, 10, lines);
    if ((int)lines.size() <= BODY_H / 16) { size = 2; return; }
    lines.clear();
  }
  wrapParagraphs(body, 21, lines);
}

// ---------------------------------------------------------------- screen (the only place that draws)
// Header row + `size`-sized text rows starting at row `first`.
void drawFrame(const String& header, const std::vector<String>& lines, int size, int first) {
  if (!haveDisplay) return;
  display.clearDisplay();
  display.setTextWrap(false);
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print(header);
  if (WiFi.status() == WL_CONNECTED) {                  // signal strength on the right
    String r = String(WiFi.RSSI()) + "dBm";
    display.setCursor(SCREEN_W - 6 * r.length(), 0);
    display.print(r);
  }
  if (SCREEN_H >= 64) display.drawFastHLine(0, 9, SCREEN_W, SSD1306_WHITE);
  display.setTextSize(size);
  int maxLines = BODY_H / (8 * size);
  for (int i = 0; i < maxLines && first + i < (int)lines.size(); i++) {
    display.setCursor(0, BODY_TOP + i * 8 * size);
    display.print(lines[first + i]);
  }
  display.display();
}

// Simple message screen (size 1).
void showMessage(const String& header, const String& msg) {
  statusLine = msg;
  Serial.println("[" + header + "] " + msg);
  std::vector<String> lines;
  wrapParagraphs(msg, 21, lines);
  drawFrame(header, lines, 1, 0);
}

// The normal screen. animate = play all pages of a long sentence once (ends on the last page).
void drawScreen(bool animate) {
  String header, body;
  bool allowBig = false, tail = false;
  if (WiFi.status() != WL_CONNECTED) {
    header = "NO WIFI";  body = statusLine;
  } else if (lastCode < 0 || lastCode >= 400) {
    header = "NO SERVER"; body = "Can't reach server\n" + serverShort() + "\nrun.py on? code " + String(lastCode);
  } else if (cur.live.length() > 0) {
    header = "SIGNING"; body = cur.live; tail = true;            // live words change fast: always show the END
  } else if (cur.text.length() > 0) {
    header = "SENTENCE #" + String(cur.id); body = cur.text; allowBig = true;
  } else {
    header = "READY"; body = "Waiting for signs...\nIP " + WiFi.localIP().toString();
  }
  Serial.printf("[%s] %s\n", header.c_str(), body.c_str());
  if (!haveDisplay) return;

  std::vector<String> lines;
  int size;
  layoutText(body, allowBig, lines, size);
  int maxLines = BODY_H / (8 * size);
  int n = lines.size();
  int pages = (n + maxLines - 1) / maxLines;
  if (pages < 1) pages = 1;
  if (tail) {                                           // last `maxLines` rows only
    drawFrame(header, lines, size, n > maxLines ? n - maxLines : 0);
  } else if (animate && pages > 1) {
    for (int p = 0; p < pages; p++) {
      drawFrame(header, lines, size, p * maxLines);
      if (p < pages - 1) delay(PAGE_MS);
    }
  } else {                                              // refresh: show the last page
    drawFrame(header, lines, size, (pages - 1) * maxLines);
  }
}

// ---------------------------------------------------------------- Wi-Fi + server
// Shown and printed on the FIRST connection and after every reconnection.
void announceConnection() {
  String ip = WiFi.localIP().toString();
  Serial.println();
  Serial.println("=========== Wi-Fi connected ===========");
  Serial.println("SSID    : " + String(WIFI_SSID));
  Serial.println("ESP32 IP: " + ip);
  Serial.println("Gateway : " + WiFi.gatewayIP().toString());
  Serial.println("Signal  : " + String(WiFi.RSSI()) + " dBm");
  Serial.println("MAC     : " + WiFi.macAddress());
  Serial.println("Server  : " + String(SERVER));
  Serial.println("=======================================");
  showMessage("WIFI OK", "IP " + ip + "\n" + serverShort());
  delay(IP_SHOW_MS);
}

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);                 // lower latency
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  unsigned long t0 = millis();
  int dots = 0;
  while (WiFi.status() != WL_CONNECTED) {
    String d = "";
    for (int i = 0; i < dots % 4; i++) d += ".";
    showMessage("WIFI", String("Connecting to ") + WIFI_SSID + d);
    delay(500);
    dots++;
    if (millis() - t0 > 20000) {        // try again from scratch
      showMessage("WIFI", "Failed. Check name/password (2.4 GHz). Retrying...");
      WiFi.disconnect(true);
      delay(1500);
      WiFi.begin(WIFI_SSID, WIFI_PASS);
      t0 = millis();
    }
  }
  announceConnection();
}

// One long-poll request. Returns the HTTP status (200 new data, 204 nothing new, negative = network error).
int pollServer(String& body) {
  HTTPClient http;
  String url = String(SERVER) + "/esp?after=" + String(afterSeq) + "&timeout=20";
  http.setConnectTimeout(4000);
  http.setTimeout(26000);               // a little longer than the server's 20 s wait
  if (!http.begin(url)) return -100;
  int code = http.GET();
  if (code == 200) body = http.getString();
  http.end();
  return code;
}

// Look for the OLED on the I2C bus: print every device found, return the address to use (0 = none).
uint8_t findOled() {
  uint8_t found[8];
  int nFound = 0;
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0 && nFound < 8) found[nFound++] = a;
  }
  Serial.print("I2C devices found:");
  if (nFound == 0) Serial.print(" none");
  for (int i = 0; i < nFound; i++) Serial.printf(" 0x%02X", found[i]);
  Serial.println();
  for (int i = 0; i < nFound; i++) if (found[i] == OLED_ADDR) return OLED_ADDR;
  for (int i = 0; i < nFound; i++) if (found[i] == 0x3C || found[i] == 0x3D) return found[i];
  return 0;
}

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("\nESP32 sign display starting");
  if (LED_PIN >= 0) pinMode(LED_PIN, OUTPUT);
  Wire.begin(OLED_SDA, OLED_SCL);
  uint8_t addr = findOled();
  haveDisplay = addr != 0 && display.begin(SSD1306_SWITCHCAPVCC, addr);
  if (haveDisplay) Serial.printf("OLED ready at 0x%02X (%dx%d)\n", addr, SCREEN_W, SCREEN_H);
  else Serial.println("OLED not found: check wiring (SDA/SCL/VCC/GND). Continuing with Serial only.");
  showMessage("SIGN DISPLAY", "Starting...");
  connectWifi();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) { lastCode = 0; connectWifi(); }

  String body;
  int code = pollServer(body);
  lastCode = code;

  if (code == 200) {
    State next = cur;
    bool ok = parseState(body, next);
    if (ok) {
      bool newSentence = next.id != cur.id && next.id > 0;
      cur = next;
      afterSeq = cur.seq;
      if (newSentence && LED_PIN >= 0) { digitalWrite(LED_PIN, HIGH); delay(120); digitalWrite(LED_PIN, LOW); }
    }
    drawScreen(true);
  } else if (code == 204) {
    drawScreen(false);                  // nothing new: refresh header / signal strength
  } else {
    drawScreen(false);                  // error screen
    delay(2000);                        // do not hammer a dead server
  }
}
