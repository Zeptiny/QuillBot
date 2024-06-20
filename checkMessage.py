import aiohttp
from responses.errors import responses


def checkMessage(message_content):
    for key in responses.keys():
        if key.lower() in message_content.lower():
            return responses[key]

async def readFileContent(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.text()
    return None

async def uploadMclogs(fileContent):
    upload_url = 'https://api.mclo.gs/1/log'
    async with aiohttp.ClientSession() as session:
        payload = {'content': fileContent}
        async with session.post(upload_url, data=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('url')
    return None
