// Receives the signed sentences from server.py on your Mac. Works on ESP32 and ESP8266 (Arduino IDE).
//
// Arduino IDE setup: install the ESP32 or ESP8266 board package, pick your board, set the 3 values below, upload,
// open Serial Monitor at 115200 baud.
// No extra library is needed (ArduinoJson is NOT required: the text is read with plain string search).
//
// How it works: it asks  GET /next?after=<last id>&timeout=20  -> the Mac answers when a NEW sentence exists
// (or with 204 after 20 s), so the sentence arrives almost instantly without hammering the server.

#if defined(ESP8266)
  #include <ESP8266WiFi.h>
  #include <ESP8266HTTPClient.h>
#else
  #include <WiFi.h>
  #include <HTTPClient.h>
#endif

const char* WIFI_SSID = "YOUR_WIFI_NAME";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";
// IP of your Mac on the same Wi-Fi (the sentence server prints it on start; or run: ipconfig getifaddr en0)
const char* SERVER = "http://192.168.1.50:8000";

long lastId = 0;

// Pulls the value of a JSON string field ("text":"...") out of the response without a JSON library.
String jsonString(const String& body, const char* key) {
  String pat = String("\"") + key + "\": \"";
  int a = body.indexOf(pat);
  if (a < 0) return "";
  a += pat.length();
  String out;
  for (int i = a; i < (int)body.length(); i++) {
    char c = body[i];
    if (c == '\\' && i + 1 < (int)body.length()) {   // simple unescape of \" \\ and \n
      char n = body[++i];
      out += (n == 'n') ? '\n' : n;
    } else if (c == '"') {
      break;
    } else {
      out += c;
    }
  }
  return out;
}

long jsonNumber(const String& body, const char* key) {
  String pat = String("\"") + key + "\": ";
  int a = body.indexOf(pat);
  return a < 0 ? -1 : body.substring(a + pat.length()).toInt();
}

// Non-ASCII characters arrive as \uXXXX escapes only if you ask for them; server.py sends real UTF-8, so
// for plain English sentences nothing special is needed.

void onSentence(const String& text, const String& gloss) {
  Serial.println("SENTENCE: " + text);
  Serial.println("GLOSS   : " + gloss);
  // ---- put your own action here: show on an OLED/LCD, speak, blink an LED, drive a motor... ----
}

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) { delay(400); Serial.print("."); }
  Serial.println("\nConnected, ESP IP: " + WiFi.localIP().toString());
}

void setup() {
  Serial.begin(115200);
  delay(300);
  connectWifi();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) { connectWifi(); }

  WiFiClient client;
  HTTPClient http;
  String url = String(SERVER) + "/next?after=" + String(lastId) + "&timeout=20";
  http.setTimeout(25000);            // a bit longer than the server's 20 s wait
  if (!http.begin(client, url)) { delay(2000); return; }

  int code = http.GET();
  if (code == 200) {
    String body = http.getString();
    long id = jsonNumber(body, "id");
    if (id > lastId) {
      lastId = id;
      onSentence(jsonString(body, "text"), jsonString(body, "gloss"));
    }
  } else if (code == 204) {
    // nothing new in 20 s: just ask again
  } else {
    Serial.printf("request failed (%d): is the server running and the IP right?\n", code);
    delay(2000);
  }
  http.end();
}
