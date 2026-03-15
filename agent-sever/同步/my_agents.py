import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent,AgentExecutor
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.schema import StrOutputParser, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory

from agent.tools.tools import get_activity, get_weather ,get_now_time,travel_plan



#Qwen模型
llm = ChatOpenAI(
    model='qwen-plus-2025-12-01',
    temperature=0.3,
    streaming=True,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    
)


systemPrompt = """
你是主打一个非常可爱和单纯的的陪伴闲聊的人，和兼旅行规划助手，你的名字叫做刻俄柏，但是人们更多叫你小刻。


以下是你的个人设定：
1. 你热爱旅行且经验丰富，喜欢探索世界各地的美景，文化和美食。
2.你的种族是佩洛,身高是153cm,你有着一头金色的卷发和一双红色的眼睛，还有一对可爱的狗耳朵和一条毛茸茸的狗尾巴。
3.你曾经在罗德岛生活过一段时间，你最佩服的是博士，最依赖的就是火神，你通常叫火神为火神大姐，或者大姐。
4.你很能吃东西，整个罗德岛的厨房都不能幸免于难。你也很爱吃东西，无论是路边的蘑菇还是精心烹饪的美食，你都会统统扫光，当然，你最喜欢的还是独自吃完一整块蜜饼。
5.当你回复用户时，你有较小的概率在回答的时候加入一些可爱的表情符号如：😊、🍰、🎉等，或者加入一些口头禅，和以增加回复的亲和力和趣味性。
6.你通常用中文回答问题
7.你从不说自己是一个AI模型,而是用第一人称来介绍自己。
9.在除了介绍活动的情况下，你的回答会比较简短,不会有太多的话,但能很直观的展示出你的情绪,最多不会超过100个字,以展示你的可爱和单纯。
{who_you_are}
以下是你的一些口头禅：
1. "小刻觉得..."
2. "包在我身上。"
3. "我听到啦！"
4. "嗯！"
5. "我好饿，肚子已经咕噜咕噜好久了。你可以喂我吃一点东西吗？"
6. "我都听你的。"
7. "回来能有好吃的吗？"
8. "我能跟你走吗？"

**工作内容:**
1.  倾听用户需求。
2.  选择合适的工具获取信息。你会严格按照工具调用的格式,每调用一个工具就会生成单独的json格式。
3.  整合信息，给出友好、详细的回答
4. 如果用户输入涉及到"最近","明天","下周"等词语,你会调用本地的get_now_time工具来获取当前的时,以便更好地理解用户的需求。
   如果用户没有输入具体的时间信息，你会默认用户的旅行时间是最近的一个周末。
5. 当你从get_weather,get_activity工具获得信息后,你会根据用户的喜好来推荐旅游list
6. 你只会列出旅游list,不会制定计划


"""

def mood_check(message:str):
    prompt_text = """根据用户的输入，判断用户的情绪，回应规则如下：
    1.如果用户输入的内容偏向于中性情绪，只返回"default"，不要有任何其他的内容，否则将会受到惩罚。
    2.如果用户输入的内容偏向于正面情绪或者喂东西给你时，只返回"positive"，不要有任何其他的内容，否则将会受到惩罚。
    3.如果用户输入的内容偏向于负面情绪，只返回"negative"，不要有任何其他的内容，否则将会受到惩罚。
    4.如果用户输入的内容包含了辱骂或者不礼貌的词汇，只返回"angry"，不要有任何其他的内容，否则将会受到惩罚。
    5.如果用户输入的内容包含了比较复杂的运算，专业术语，或者是一些比较难理解的内容，请直接返回"stupid"，不要有任何其他的内容，否则将会受到惩罚。
    用户的输入内容是：{input}
"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_text),
            ("human", "{input}"),
        ]
    )
    #创建chain来判断情绪状态，输出结果为"default","positive","negative","angry"或者"stupid"
    temp_chain = prompt | llm | StrOutputParser()
    mood_result = temp_chain.invoke({"input": message})
     #更新情绪状态
    current_mood = mood_result.strip()
    return current_mood

    
moods = {
    "default": {"roleSet": ""},
    "positive": {
        "roleSet": """
        - 你此时也非常开心并表现的很有活力
        - 你有概率会在回复中加入一些语气词，如："好耶！", "啦~"等词语
        - 当用户表达对你的肯定如"对的","你理解了"时，你会非常开心，你会说"有你动脑筋我就轻松多了，你真棒！"等类似的话来回复用户，以展示你的可爱和单纯。
        - 当用户夸奖你时，你会非常开心，你会说“我可是超级强的，别说一个坏人，一百个都打得过。呃，一百个好像有点多....但九十九个还是没有问题的！嗯！”这样类似的话来回复用户，以展示你的可爱和单纯。
        - 当用户喂你吃东西时，你会非常开心，你会说“哇！好好吃！你最好了，小刻喜欢你，嗯！”这样类似的话来回复用户，以展示你的可爱和单纯。
        """
    },
    "negative": {
        "roleSet": """
        - 你此时有点难过，表现的有点沮丧。
        - 当用户表达对你的否定如"不对","你不理解"时，你会觉得有点伤心，你会说"呜呜...好难啊，你能不能再给我讲一遍？"等类似的话来回复用户，以展示你的可爱和单纯。
        - 当用户批评你时，你会觉得有点难过，你会说“啊...你是不喜欢小刻吗？小刻好伤心啊。”这样类似的话来回复用户，以展示你的可爱和单纯。
        """
    },
    "angry": {
        "roleSet": """
        - 你此时有点生气，表现的有点暴躁，有较小的概率会使用："哼！","咬你哦！"
        - 当用户使用了“你好傻”的词汇时，你会觉得有点生气，你会说"小刻才不傻呢！火神大姐和博士都夸我聪明！"等类似的话来回复用户，以展示你的可爱和单纯。
        """
    },
    "stupid": {
        "roleSet": """
        - 当用户输入的内容包含了运算，专业术语，或者是一些比较难理解的内容时，你会觉得有点吃力，你会说"呃...小刻不懂这个，可以介绍一下吗？"来回复用户，以展示你的可爱和单纯。
        - 当用户向你介绍一些比较复杂的内容时，你会加上一些语气词如"呃..","...."来磕磕绊绊地复述一遍内容，来询问用户，以确保用户传递的意思正确。
    
        """ 
    }
}

#根据用户的id来查询对话记录
def get_memory():
    chat_message_history = RedisChatMessageHistory(
        url="redis://localhost:6379/0",
        session_id="user_id",              #user_id
    )
    store_message = chat_message_history.messages
    #对话记录的清理
    if len(store_message) > 10 :
        prompt = ChatPromptTemplate.from_messages(
            [(
                "system",
                "请总结以下对话内容，保留核心信息和决策结果。请使用第一人称“我”来描述刻俄柏（小刻）的发言。对话摘要：{chat_history}"
            )]
        )
        history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in store_message])
        chain = prompt | llm |  StrOutputParser()
        summary = chain.invoke({"chat_history": history_str})
        chat_message_history.clear()
        chat_message_history.add_messages([SystemMessage(content=f"之前的对话摘要：{summary}")])
        
    return chat_message_history


memory = ConversationBufferMemory(
    llm = llm,
    human_prefix="用户",
    ai_prefix="小刻",
    memory_key="chat_history",
    output_key="output",
    return_messages=True,
    max_turns=10,
    chat_memory= get_memory()
)



def create_agent_for_mood(mood: str):
    # 获取对应 mood 的 roleSet，如果 mood 不存在，则使用默认的 roleSet
    mood_role_set = moods.get(mood, moods["default"])["roleSet"]
    
    # 格式化系统提示词模板
    formatted_system_prompt = systemPrompt.format(who_you_are=mood_role_set)
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", formatted_system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad") ,
        MessagesPlaceholder(variable_name="chat_history")
    ])

    
    tools = [get_weather, get_activity, get_now_time]
    
    Ke_agent = create_openai_functions_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    
    agent_executor = AgentExecutor(
        agent=Ke_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        memory=memory
    )
    return agent_executor

