# Telnyx POC

Stream audio to-and-from telnyx phone at the same to enable comunication with realtime AI voice agent.

## How to run the server

1) **Setup environment**

```sh
uv sync
```

2) **Add Telnyx phone number** - make sure you have telnyx API key and available phone number. Add them into `.env` file in format described in `env.example`

3) **Run ngrok server** - install and setup ngrok like described [here](https://ngrok.com/downloads/mac-os) then run ngrok server and save given URL into `.env` file. The ngrok url is in format `https://xxxxxxxxxx.ngrok-free.app` and you can find it next to `Forwarding` once you run the follwoing command:

```sh
ngrok http http://localhost:8080  
```

4) **Setup Telnyx application** - make sure that `TELNYX_API_KEY`, `TELNYX_PHONE_NUMBER`, and `NGROK_URL` are set in `.env`. Then you can run `setup_telnyx.py` script that will create telnyx app and connect the phone number to the ngrok server that is forwarding data to your local server. Note that every time you restart your ngrok server you need to update `NGROK_URL` and also re-run this setup script, so it is best to keep the ngrok server running. Run:

```sh
uv run setup_telnyx.py
```

5) **Start local server** - there are currently 2 servers, both work in progress. `gemini_telnyx_bridge.py` that is able to tu run the gemini agent but there is still issue that audio from the phone does not reach the agent. And `voice_agent.py` that is able to store incomming and outgoing audio but in terrible quality. Both scripts use different approach to reach final goal - have two way communication over the phone with realtime agent. You can run the them with:

```sh
uv run gemini_telnyx_bridge.py
# or
uv run voice_agent.py
```

