"""
https://huggingface.co/eachadea/ggml-vicuna-13b-1.1/blob/main/ggml-vicuna-13b-1.1-q4_2.bin
"""

import re
from langchain import Wikipedia

from langchain.llms import LlamaCpp
from langchain.agents.react.base import DocstoreExplorer
from langchain.tools import DuckDuckGoSearchRun

docstore=DocstoreExplorer(Wikipedia())
dd_search = DuckDuckGoSearchRun()


prefix = """質問に対し以下のフォーマットで回答して下さい。"""

tools_prompt = """以下のツールが選択できます
- Web検索: Webページを検索する
- Wikipedia検索: Wikipediaを検索する
"""

tools = """[Web検索]、[Wikipedia検索]"""

react_prompt = f"""以下のフォーマットで出力して下さい
- 思考: 質問に対しあなたが解決すべき課題
- ツールの選択: {tools}から行動を１つ選択して下さい
- ツールへの入力: ツールに入力するキーワード"""

react_prompt_final_answer = f"""{react_prompt}
(思考を繰り返し、ツールを選択する必要がない場合、最終的な回答を出力してください。)
- 最終的な回答: 質問に対する最終的な答えを出力して下さい。"""


def create_first_prompt(qa):
    return f"""### HUMAN: {prefix}

{tools_prompt}

{react_prompt}

{qa}
### ASSISTANT: """


def create_wiki_react_prompt(qa, info):
    return f"""{prefix}

追加情報: {info}
    
{tools_prompt}

{react_prompt_final_answer}

### HUMAN: {qa}
### ASSISTANT: """


def create_qa_prompt(qa, info):
    return f"""### Human: 関連情報を用いて、質問に答えてください。

関連情報: {info}

### HUMAN: {qa}
### ASSISTANT: """

def create_summary_prompt(info):
    return f"""
    文章: {info}

    ### HUMAN: 文章を要約してください。与えられた情報のみを利用してください。
    ### ASSISTANT: """

stop = [
    '\n### Human: ',
    '\n\t### Human: ',
    '\nHUMAN: ',
    '\n\tHUMAN: ',
]
model_path="../llama.cpp/models/ggml-vicuna-13b-4bit.bin"
model_path="../llama.cpp/models/ggml-vicuna-13b-1.1-q4_2.bin"
model_path='../llama.cpp/models/ggml-vic13b-q5_1.bin'

llm = LlamaCpp(
    model_path=model_path,
    n_ctx=1024,
    max_tokens=256,
    stop=stop,
    streaming=False
)


def get_tool(t):
    action = None
    regex = r"ツールの選択[: |：].*?\[(.*?)\].*?[\n]+"
    match = re.search(regex, t, re.DOTALL)
    if match:
        action = match.group(1)

    action_input = None
    regex = r"ツールへの入力[: |：](.*?)[\n]+"
    match = re.search(regex, t, re.DOTALL)    
    if match:        
        action_input = match.group(1)

    return action, action_input

def get_final_answer(t):
    regex = r"最終的な回答[: |：](.*?)[\n]+"
    match = re.search(regex, t, re.DOTALL)
    if match:
        result = match.group(1)
        return result
    return None


qa='日本の現在の総理大臣を教えてください' 


## ツールの選択/ツールへの入力
while True:
    prompt = create_first_prompt(qa)
    output = llm(prompt)
    output = output + '\n'

    action, action_input = get_tool(output)

    if tool_result != None and tool_result != '':
        tool_result = None
        if action == 'Web検索':
            tool_result = dd_search.run(action_input.strip())
            tool_result.strip()
        elif action == 'Wikipedia検索':
            tool_result = docstore.search(action_input.strip())
            tool_result.strip()
        else:
            tool_result = None

        if tool_result != None and tool_result != '':
            tool_result = tool_result[:250]
            break

            ## summary
            # tool_prompt = create_summary_prompt(tool_result[500:])
            # tool_result = llm(tool_prompt)
            # print('tool_result summary', tool_result)
            # print('--------------------')

prompt = create_wiki_react_prompt(qa, tool_result)
while True:
    output = llm(prompt)
    output = output + '\n'
    final_answer = get_final_answer(output)
    if final_answer:
        break

qa_prompt = create_qa_prompt(qa, final_answer)
result = llm(qa_prompt)
print('result: ', result)


