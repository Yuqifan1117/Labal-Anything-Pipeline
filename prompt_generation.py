from get_token_ids import get_token_ids_for_task_parsing, get_token_ids_for_choose_model, count_tokens, get_max_context_length
import requests
import json

config = {}    
config["openai_key"] = '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'
config["model"] = "gpt-3.5-turbo"
config["use_completion"] = False
use_completion = config["use_completion"]
LLM = config["model"]
# /v1/chat/completions	gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301	
# /v1/completions	text-davinci-003, text-davinci-002, text-curie-001, text-babbage-001, text-ada-001, davinci, curie, babbage, ada
if LLM == "gpt-3.5-turbo":
    LLM_encoding = "text-davinci-003"
OPENAI_KEY = config["openai_key"]
url = "https://api.openai.com/v1/chat/completions"
HEADER = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_KEY}"
}
PROXY = {
    "https": '127.0.0.1:7890',
}

def convert_chat_to_completion(data):
    messages = data.pop('messages', [])
    tprompt = ""
    if messages[0]['role'] == "system":
        tprompt = messages[0]['content']
        messages = messages[1:]
    final_prompt = ""
    for message in messages:
        if message['role'] == "user":
            final_prompt += ("<im_start>"+ "user" + "\n" + message['content'] + "<im_end>\n")
        elif message['role'] == "assistant":
            final_prompt += ("<im_start>"+ "assistant" + "\n" + message['content'] + "<im_end>\n")
        else:
            final_prompt += ("<im_start>"+ "system" + "\n" + message['content'] + "<im_end>\n")
    final_prompt = tprompt + final_prompt
    final_prompt = final_prompt + "<im_start>assistant"
    data["messages"] = messages
    data['stop'] = data.get('stop', ["<im_end>"])
    data['max_tokens'] = data.get('max_tokens', max(get_max_context_length(LLM) - count_tokens(LLM_encoding, final_prompt), 1))
    return data
def replace_slot(text, entries):
    for key, value in entries.items():
        if not isinstance(value, str):
            value = str(value)
        text = text.replace("{" + key +"}", value.replace('"', "'").replace('\n', ""))
    return text
def send_request(data):
    if use_completion:
        data = convert_chat_to_completion(data)
    # response = requests.post(url, json=data, headers=HEADER, proxies=PROXY)
    # print(response.json())
    try:
        response = requests.post(url, headers=HEADER, data=json.dumps(data), proxies=PROXY)
        response_json = json.loads(response.text)
        if "choices" in response_json and len(response_json["choices"]) > 0:
            if use_completion:
                return response.json()["choices"][0]["text"]
            else:
                return response.json()["choices"][0]["message"]["content"]
        else:
            print(response_json)
            return response_json
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    return None
    

if __name__ == '__main__':
    # chatgpt as descriptor to provide more visual features for specific label word
    visual_feature_dict = dict()
    # label_words = ['red panda','Siberian Husky','Alaska dog']
    label_words = json.load(open('self_instruction/label_words.json'))
    for k in label_words:
        label_word = label_words[k]
        messages = json.loads(open('self_instruction/descriptor_demo_prompt.json',"r").read())
        response_results_tprompt = "You must first consider the important visual features related to the input topic and present them in a list. Consider these keywords step by step based on the visual features of the topic and generate the appropriate prompts for midjourney v5. Then please detailedly elaborate on your workflow, including the visual features used and all intermediate annotation results. If annotated information for images, audio or video of any file is generated in the inference result, must tell me the complete path. If there is nothing in the results, please tell me you can't make it."
        prompt_template = open('self_instruction/prompt_template/descriptor_prompt_template.txt').read()
        description_prompt = replace_slot(prompt_template, {
            "category name": label_word
        })
        messages.insert(0, {"role": "system", "content": response_results_tprompt})
        messages.append({"role": "user", "content": description_prompt})
        # logger.debug(messages)
        data = {
            "model": LLM,
            "messages": messages,
            "temperature": 0
        }
        print(description_prompt)
        results = send_request(data)
        retries = 0
        while 'error' in results and retries < 3:
            results = send_request(data)
            retries += 1
        
        # results = "- reddish-brown fur\n- white face with tear markings\n- black nose- round ears\n- bushy tail with alternating red and white rings\n- short legs\n- white paws with black claws"
        if 'error' not in results:
            visual_feature_set = results.replace('- ',"").split('\n')
            visual_feature_dict[label_word] = visual_feature_set
    json.dump(visual_feature_dict, open('self_instruction/prompt_template/coco_visual_features.json','w'))
    ###test
    visual_feature_dict = json.load(open('self_instruction/prompt_template/visual_features.json'))
    # chatgpt as a educator to guide AIGC models for various image generation contrained by visual features
    template = 'Give me 10 high quality prompt for stunning close-up photorealistic illustration of {label word} for text-to-image models with visual features of {visual features}.'
    for k in visual_feature_dict:
        generation_prompt = replace_slot(template, {
            "label word": k,
            "visual features": ', '.join(visual_feature_dict[k])
        })

    generation_messages = json.loads(open('self_instruction/prompt_template/assisdant_sd_prompt.json',"r").read())
    generation_messages.append({"role": "user", "content": generation_prompt})
    # logger.debug(messages)
    generation_data = {
        "model": LLM,
        "messages": generation_messages,
        "temperature": 0
    }
    results = send_request(generation_data)
    json.dump(results, open('self_instruction/prompt_template/generation_results.json','w'))
    print(results)
    # chatgpt as a creator to edit and splice each object-centric image, thereby composing scenes of complex object interactions
    # Also, imagine various scenarios that exist in the real world as background scenes
