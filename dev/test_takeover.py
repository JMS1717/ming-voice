import asyncio, json, websockets
async def main():
    a = await websockets.connect("ws://localhost:8000/ws")
    print("A hello:", json.loads(await a.recv())["type"])
    b = await websockets.connect("ws://localhost:8000/ws")
    print("B hello:", json.loads(await b.recv())["type"])
    try:
        msg = await asyncio.wait_for(a.recv(), 5)
        print("A got:", json.loads(msg)["type"])
        await asyncio.wait_for(a.recv(), 5)
    except websockets.ConnectionClosed as e:
        print("A closed with code", e.rcvd.code if e.rcvd else None)
    await b.send(json.dumps({"type": "text", "text": "Say OK."}))
    while True:
        m = await asyncio.wait_for(b.recv(), 30)
        if isinstance(m, str) and json.loads(m)["type"] == "done":
            print("B reply:", json.loads(m)["text"]); break
    await b.close()
asyncio.run(main())
