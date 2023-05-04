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

# - 観察: 行動により得られた結果から質問の回答に必要な情報を抽出して下さい"""

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


def create_qa_prompt(qa, info, history):
    return f"""### Human: 関連情報を用いて、質問に答えてください。

関連情報: {info}

{history}
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

def to_hisotry(qa, final_result):
    final_result = final_result[:150]

    return f"""
### HUMAN: {qa}
### ASSISTANT: {final_result}    
"""

def check_output(qa):
    prompt = create_first_prompt(qa)

    print('prompt: ', prompt)
    output = llm(prompt)
    print('output: ', output)
    exit(0)

def check_summary():
    info = """この項目では、日本の内閣総理大臣の一覧について説明しています。. その他の用法については「 内閣総理大臣 (曖昧さ回避) 」をご覧ください。. 内閣総 理大臣. 現職. 岸田文雄. 第2次岸田改造内閣. 就任日：2021年11月10日. 歴代の首相と内閣. 歴代内閣総理大臣. 内閣総理大臣の一覧 （ないかくそうりだいじんのいちらん）は、 日本 の 行政府の長 である 内閣総理大臣 を務めた人物の一覧である。. 脚注. [ 続きの解説] 「内閣総理大臣の一覧」の続きの解説一覧. 1 内閣総理大臣の一覧とは. 2 内閣総理大臣の一覧の ... この項目では、日本の内閣総理大臣の一覧について説明しています。. その他の用法については「 内閣総理大臣 (曖昧さ回避) 」をご覧ください。. 内閣総理大臣. 現職. 岸田文雄. 第2次岸田改造内閣. 就任日：2021年11月10日. 歴代の首相と内閣. 歴代内閣総理大臣. 内閣総理大臣 （ないかくそうりだいじん、 英: Prime Minister [1] ）は、 日本 の 内閣 の 首長 たる 国務大臣 [2] 。. 文民 である 国会議員 が就任し、その 地位 及び 権限 は 日本国憲法 や 内閣法 などに規定されている [3] 。. 脚注. [ 続きの解説] 「内閣総理大臣 ... 内閣総理大臣 （ないかくそうりだいじん、 英: Prime Minister [1] ）は、 日本 の 内 閣 の 首長 たる 国務大臣 [2] 。. 文民 である 国会議員 が就任し、その 地位 及び 権限 は 日本国憲法 や 内閣法 などに規定されている [3] 。. 脚注. [ 続きの解 説] 「内閣総理大臣"""
    print(len(info))
    info = info[500:]
    p = create_summary_prompt(info)
    r = llm(p)
    print('r', r)
    exit(0)

# qa='埼玉県で一番高い山を教えてください' 
qa='日本の現在の総理大臣を教えてください' 
#qa='kubernetesのpodについて教えてください' 
# qa='kubernetesのserviceとpodの関係について教えてください' 
# qa='LLMのReActについて教えてください' 

# check_output(qa)
# check_summary()

while False:
    while True:
        prompt = create_first_prompt(qa)
        print('prompt', prompt)
        output = llm(prompt)
        print('first result: ', output)
        print("==========")
        output = output + '\n'

        action, action_input = get_tool(output)
        
        print('action:', action)
        print('actoin_input:', action_input)
        print("==========")
        if action and action_input:
            break

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
        print('tool_result', tool_result)
        print('--------------------')
        # tool_prompt = create_summary_prompt(tool_result[500:])
        # tool_result = llm(tool_prompt)
        # print('tool_result summary', tool_result)
        # print('--------------------')

        prompt = create_wiki_react_prompt(qa, tool_result)
        print('react prompt', prompt)
        print('--------------------')
        while True:
            output = llm(prompt)
            print('react result: ', output)
            print('--------------------')
            output = output + '\n'
            print("==========")
            final_answer = get_final_answer(output)
            if final_answer:
                break

# pre_qa = "kubernetesのpodについて教えてください"
# pre_ans = """こんにちは！KubernetesのPODについて説明します。

# Kubernetesでは、1つのクラスター内の複数のノードに、1つのポッドを配置することができます。このポッドは、コンテナー内で実行されるアプリケーションやプログラムをカスタマイズしたものになります。

# 各ポッドは、Kubernetesクラスター内の1つのノードに分けられます。このようにすることで、ポッドが互いに独立して実行されるため、1つのアプリケーションの失効や1つのノー ドの挫折による制限を回避することができます。

# また、ポッドは簡単に動的なイ"""
# history = to_hisotry(pre_qa, pre_ans)

#final_answer = '現在の内閣総理大臣は岸田文雄です。'
#qa_prompt = create_qa_prompt(qa, final_answer, "")

qa_prompt="""### HUMAN: 日本の現在の総理大臣を教えてください
### ASSISTANT:
"""
print('qa prompt:', qa_prompt)
print('--------------')
result = llm(qa_prompt)
print('result: ', result)


