import json
import requests

class VllmGPT:

    def __init__(self, host="192.168.1.3",
                 port="8101",
                 model="THUDM/chatglm3-6b",
                 max_tokens="1024"):
        self.host = host
        self.port = port
        self.model=model
        self.max_tokens=max_tokens
        self.__URL = f"http://{self.host}:{self.port}/v1/completions"
        self.__URL2 = f"http://{self.host}:{self.port}/v1/chat/completions"

    def _send_request(self, url, content):
        headers = {'content-type': 'application/json'}
        r = requests.post(url, headers=headers, json=content)
        return json.loads(r.text)

    def chat(self,cont):
        chat_list = []
        content = {
            "model": self.model,
            "prompt":"Please give me a brief reply" +  cont,
            "history":chat_list}
        res = self._send_request(self.__URL, content)
        return res['choices'][0]['text']

    def question2(self,cont):
        chat_list = []
        content = {
            "model": self.model,
            "prompt":"Please give me a brief reply" +  cont,
            "history":chat_list}
        res = self._send_request(self.__URL2, content)
        return res['choices'][0]['message']['content']

if __name__ == "__main__":
    vllm = VllmGPT('192.168.1.3','8101')
    req = vllm.chat("your question here.")
    print(req)