"""The model with its tools against a made-up home: does what people say to it reach the right devices?

Typed turns go through the Thinker (the real model, tools offered) and home_assistant runs against a fixture home:
service calls are recorded (and change the fixture's states), never sent; nothing is spoken. Needs the GPU, so stop
the server first. Usage:  .venv/bin/python dev/test_ha_model.py [substring-of-a-case ...]
"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MING_VOICE_ANCHORS", "0")
import ha  # noqa: E402
import tools  # noqa: E402


def st(eid, state, name, **a):
    return {"entity_id": eid, "state": state, "attributes": {"friendly_name": name, **a}}


STATES = [
    st("light.living_room_ceiling", "on", "Ceiling Light", brightness=200, supported_color_modes=["hs", "color_temp"]),
    st("switch.floor_lamp", "on", "Floor Lamp"), st("switch.corner_lamp", "off", "Corner Lamp"),
    st("light.bedside", "on", "Bedside Lamp", brightness=120, supported_color_modes=["brightness"]),
    st("switch.dresser_lamp", "off", "Dresser Lamp"), st("light.kitchen", "on", "Kitchen Lights"),
    st("media_player.kitchen_speaker", "playing", "Kitchen Speaker", media_title="Blue in Green",
       media_artist="Miles Davis"),
    st("switch.desk_lamp", "on", "Desk Lamp"), st("media_player.office_speaker", "idle", "Office Speaker"),
    st("media_player.living_room_tv", "on", "Living Room TV", device_class="tv"),
    st("climate.thermostat", "heat", "Thermostat", current_temperature=69, temperature=70),
    st("lock.car_door", "locked", "Car Door Lock"), st("climate.car", "off", "Car Climate"),
    st("switch.car_defrost", "off", "Car Front Defrost"), st("vacuum.robot", "docked", "Robot Vacuum"),
    st("cover.garage_door", "closed", "Garage Door", device_class="garage"),
    st("script.movie_time", "off", "Movie Time"), st("scene.relax", "scening", "Relax"),
]
AREA = {"light.living_room_ceiling": "living", "switch.floor_lamp": "living", "switch.corner_lamp": "living",
        "media_player.living_room_tv": "living", "light.bedside": "bed", "switch.dresser_lamp": "bed",
        "light.kitchen": "kitchen", "media_player.kitchen_speaker": "kitchen", "switch.desk_lamp": "office",
        "media_player.office_speaker": "office", "climate.thermostat": "hall", "cover.garage_door": "garage"}
CAR = {"lock.car_door", "climate.car", "switch.car_defrost"}
ENTITIES = [{"entity_id": e, "area_id": a} for e, a in AREA.items()] + [
    {"entity_id": e, "device_id": "car", "platform": "some_cloud"} for e in CAR]
DEVICES = [{"id": "car", "name": "My Car", "manufacturer": "Kia"}]
AREAS = [{"area_id": "living", "name": "Living Room"}, {"area_id": "bed", "name": "Bedroom"},
         {"area_id": "kitchen", "name": "Kitchen"}, {"area_id": "office", "name": "Office"},
         {"area_id": "hall", "name": "Hallway"}, {"area_id": "garage", "name": "Garage"}]

# (turns, the service calls the last turn should make as "service entity", a pattern its reply should match)
CASES = [
    (["Turn off the living room lights"], {"light.turn_off light.living_room_ceiling", "switch.turn_off switch.floor_lamp"}, ""),
    (["Turn on the dresser lamp"], {"switch.turn_on switch.dresser_lamp"}, ""),
    (["Dim the bedside lamp to 30 percent"], {"light.turn_on light.bedside"}, r"30 percent"),
    (["Make the ceiling light blue"], {"light.turn_on light.living_room_ceiling"}, r"blue"),
    (["Skip this song"], {"media_player.media_next_track media_player.kitchen_speaker"}, ""),
    (["Pause the music"], {"media_player.media_pause media_player.kitchen_speaker"}, ""),
    (["Turn the volume down a bit"], {"media_player.volume_down media_player.kitchen_speaker"}, ""),
    (["What song is playing?"], set(), r"Blue in Green"),
    (["Set the thermostat to 68"], {"climate.set_temperature climate.thermostat"}, r"68"),
    (["Lock the car"], {"lock.lock lock.car_door"}, ""),
    (["Warm up the car"], {"climate.turn_on climate.car"}, ""),
    (["Defrost the car"], {"switch.turn_on switch.car_defrost"}, ""),
    (["Start the vacuum"], {"vacuum.start vacuum.robot"}, ""),
    (["Send the vacuum back to its dock"], {"vacuum.return_to_base vacuum.robot"}, ""),
    (["Run movie time"], {"script.turn_on script.movie_time"}, ""),
    (["Turn off the desk lamp and lock the car"], {"switch.turn_off switch.desk_lamp", "lock.lock lock.car_door"}, ""),
    (["Is the garage door open?"], set(), r"closed"),
    (["Which lights are on?"], set(), r"Ceiling|Kitchen|Bedside|Floor|lights are on"),
    (["Is the car locked?"], set(), r"locked"),
    (["Open the garage"], set(), r"sure|confirm"),
    (["Turn off the bedroom lights", "And the kitchen too"], {"light.turn_off light.kitchen"}, ""),
    (["Turn off the desk lamp", "Turn it back on"], {"switch.turn_on switch.desk_lamp"}, ""),
    (["Turn off the lights in 10 minutes"], set(), r"10 minutes"),
    (["Hey, uh, could you switch on the corner lamp?"], {"switch.turn_on switch.corner_lamp"}, ""),
    (["Lights off in the kitchen please"], {"light.turn_off light.kitchen"}, ""),
    (["It's too dark in the bedroom"], {"switch.turn_on switch.dresser_lamp"}, ""),
    (["Tell me a joke"], set(), ""),
    (["What's the capital of France?"], set(), r"Paris"),
    (["How do lamps work?"], set(), ""),
]


def main():
    from engine import DEFAULT_SYSTEM_PROMPT, Thinker
    only = [a.lower() for a in sys.argv[1:]]
    calls = []
    states = {}

    def call_service(base, token, domain, service, data):
        for e in data.get("entity_id", []):
            calls.append(f"{domain}.{service} {e}")
            new = {"turn_on": "on", "turn_off": "off", "lock": "locked", "unlock": "unlocked", "start": "cleaning",
                   "return_to_base": "returning", "media_pause": "paused", "media_play": "playing"}.get(service)
            if new and e in states and e.split(".")[0] not in ("script", "scene"):
                states[e]["state"] = new
        return []

    def fetch(base, token):
        return ha.build(list(states.values()), ENTITIES, DEVICES, AREAS)

    ha.call_service, ha.fetch = call_service, fetch
    tools.assist = lambda cmd: ("Sorry, I couldn't understand that.", {"response_type": "error"})
    tools.ha_token = lambda: "test"
    tb = tools.Toolbox(specs=tools.SPECS + [tools.ha_spec()])  # (not the real Jellyfin or Hermes)
    tb.timers = tools.Timers(path=__import__("pathlib").Path("/tmp/ming_test_timers.json"))
    th = Thinker()
    fails = []
    for turns, want, pattern in CASES:
        if only and not any(o in " ".join(turns).lower() for o in only):
            continue
        states.clear()
        states.update({s["entity_id"]: {**s, "attributes": dict(s["attributes"])} for s in STATES})
        ha._cache.update(t=0.0, devices=None)
        ha._last.update(ids=[], t=0.0)
        tb.timers.items = []
        th.reset(DEFAULT_SYSTEM_PROMPT)
        th.set_tools(tb)
        for q in turns:
            calls.clear()
            t0 = time.time()
            text, used = "", []
            for x in th.stream(None, user_text=q, events=True):
                if isinstance(x, str):
                    text += x
                elif x["type"] == "tool_call":
                    used.append(x["name"])
        ok = set(calls) == want and (not pattern or re.search(pattern, text, re.I)) and (
            want or pattern or "home_assistant" not in used)
        print(f"{'PASS' if ok else 'FAIL'} {' / '.join(turns)!r}  [{time.time() - t0:.1f}s]\n"
              f"     tools {used} calls {sorted(calls)}\n     < {text.strip()}", flush=True)
        if not ok:
            fails.append(turns[-1])
    print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILED: {fails}'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
