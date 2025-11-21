import json
from channels.generic.websocket import AsyncWebsocketConsumer

class TrainProgressConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        await self.send(json.dumps({"status": "connected"}))

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        # Just a dummy echo, actual updates will be pushed from training thread
        await self.send(text_data=json.dumps({"echo": text_data}))
