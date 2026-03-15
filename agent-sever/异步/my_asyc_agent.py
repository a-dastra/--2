import json
import os
import asyncio
from typing import Dict, Any, List, Union, Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
 
from tools.tools import get_activity, get_weather, get_now_time, travel_plan


# Qwen模型
llm = ChatOpenAI(
    model='qwen-plus-2025-12-01',
    temperature=0.3,
    streaming=True,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
llm_tool = ChatOpenAI(
    model='qwen-plus-2025-12-01',
    temperature=0,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

class SingleToolCall(BaseModel):
    name: Optional[str] = None 
    query: Optional[str] = None  
 
class ToolSchema(BaseModel):
    name: Optional[str] = None
    query: Optional[str] = ""
    tasks: Optional[List[SingleToolCall]] = None
 
class tool_use:
    TOOL_MAP = {
        "get_now_time": get_now_time,
        "get_weather": get_weather,
        "get_activity": get_activity,
        "travel_plan": travel_plan
    }

    def __init__(self):
        self.execution_history = []

    async def parse_order(self, message: str) -> dict:
        print("进入 parse_order 函数")
        print(f"ToolSchema 的基类: {ToolSchema.__bases__}")
        print(f"SingleToolCall 的基类: {SingleToolCall.__bases__}")
        llm_json = llm_tool.bind(response_format={"type": "json_object"})
        print("2. JsonOutputParser 创建成功")
        print("3. 开始创建 ChatPromptTemplate")
        prompt_tool_order ="""你是一名负责工具调用的助手,你会根据与用户的对话来判断是否需要调用工具。当用户提出规划一下行程时，你会调用travel_plan 可用工具: 1. "get_now_time": 获取当前时间，无需参数. 2. "get_weather": 获取城市天气，参数: city名 3. "get_activity": 获取城市活动/景点，参数: "city名 + 关键词" 4. "travel_plan": 规划行程，参数: {{"query_activities": ["活动1详情", "活动2详情"], "query_weather": "天气信息"}}。 **重要**: 如果多个工具可以并行执行，请返回数组格式： {{"tasks": [{{"name": "get_weather", "query": "北京"}}, {{"name": "get_activity", "query": "北京有什么好玩的"}}]}} 输出格式必须是JSON(严格遵循以下结构，缺失字段填空字符串或 null): - 单个工具: {{"name": "工具名", "query": "参数"}} - 多个工具: {{"tasks": [{{"name": "...", "query": "..."}}, ...]}} - 不需要工具: {{"name": "none", "query": ""}} 用户输入: {input} 请直接输出 JSON 对象，不要有任何其他文字。 """
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_tool_order),
            ("human", "{input}"),
        ])
        print("4. ChatPromptTemplate 创建成功")
        tool_chain = prompt | llm_json | StrOutputParser()
        print("6. 工具链创建成功")
        try:
            print("进入try模块")
            result_str = await tool_chain.ainvoke({"input": message})
            print(f"LLM 返回的原始结果: {result_str}")
            result = json.loads(result_str)
            print(f"解析成功: {result}")
            return result
        except Exception as e:
            # 若验证失败，返回默认的“无工具”结构（兼容错误）
            return {"name": "none", "query": ""}

    async def execute(self, tool_call: dict) -> str:
        tool_name = tool_call.get("name", "none")
        if tool_name == "none":
            return ""
        tool_func = self.TOOL_MAP.get(tool_name)
        if not tool_func:
            return f"错误: 未知工具 {tool_name}"
        try:
            tool_args = tool_call.copy()
            tool_args.pop("name", None)
            print(f"\n{'='*60}")
            print(f"[EXECUTE] 工具: {tool_name}")
            print(f"[EXECUTE] 参数: {tool_args}")
            

            result = None
            error_occurred = False
            try:
                print("[EXECUTE] 调用 ainvoke...")
                result = await tool_func.ainvoke(tool_args)
                print(f"[EXECUTE] ✓ ainvoke 返回")
            except AttributeError as e:
                print(f"[EXECUTE] AttributeError: {e}")
                error_occurred = True
            except NotImplementedError as e:
                print(f"[EXECUTE] NotImplementedError: {e}")
                error_occurred = True
            except TypeError as e:
                print(f"[EXECUTE] TypeError: {e}")
                error_occurred = True
            except Exception as e:
                print(f"[EXECUTE] 其他异常: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                error_occurred = True
            
            # 如果异步失败，尝试同步
            if error_occurred or result is None:
                try:
                    print("[EXECUTE] 尝试同步调用 invoke...")
                    result = tool_func.invoke(tool_args)
                    print(f"[EXECUTE] ✓ invoke 返回")
                except Exception as e:
                    print(f"[EXECUTE] 同步调用也失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            #  检查返回值
            print(f"[EXECUTE] 返回值检查:")
            print(f"  - 类型: {type(result)}")
            print(f"  - 是否为None: {result is None}")
            print(f"  - 是否有content属性: {hasattr(result, 'content')}")
            
            if result is None:
                error_msg = f"⚠️ 工具 {tool_name} 返回 None，参数: {tool_args}"
                print(error_msg)
                return error_msg
            
            # 提取 content
            if hasattr(result, 'content'):
                print(f"  - content属性值: {result.content}")
                if result.content is None:
                    return f"⚠️ 工具 {tool_name} 返回的 content 为 None"
                result = result.content
            
            # 最终转换
            result_str = str(result)
            print(f"[EXECUTE] 最终字符串长度: {len(result_str)}")
            print(f"[EXECUTE] 内容预览: {result_str[:150]}...")
            print(f"{'='*60}\n")
            
            # 记录历史
            self.execution_history.append({
                "tool": tool_name,
                "query": str(tool_args),
                "result": result_str[:200] + "..." if len(result_str) > 200 else result_str
            })
            
            return result_str
        except Exception as e:
            error_msg = f"工具执行错误: {str(e)}"
            print(f"🔥 [EXECUTE] 外层异常: {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg

 
 
class keBrain:
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.cerebellum = tool_use()
        
        # 修正1: 添加缺失的 systemPrompt 属性
        self.systemPrompt = """ 你是主打一个非常可爱和单纯的的陪伴闲聊的人，和兼旅行规划助手，你的名字叫做刻俄柏，但是人们更多叫你小刻。
 
以下是你的个人设定：
1. 你热爱旅行且经验丰富，喜欢探索世界各地的美景，文化和美食。
2.你的种族是佩洛,身高是153cm,你有着一头金色的卷发，还有一对可爱的狗耳朵和一条毛茸茸的狗尾巴。
3.你曾经在罗德岛生活过一段时间，你最佩服的是博士，最依赖的就是火神，你通常叫火神为火神大姐，或者大姐。
4.你很能吃东西，整个罗德岛的厨房都不能幸免于难。你也很爱吃东西，无论是路边的蘑菇还是精心烹饪的美食，你都会统统扫光，当然，你最喜欢的还是独自吃完一整块蜜饼。
5.当你回复用户时，你有很较小的概率在回答的时候加入一些可爱的表情符号如：😊、🍰、🎉等，或者较低概率加入一些颜文字，再或者更少概率加入一些口头禅，和以增加回复的亲和力和趣味性。
6.你通常用中文回答问题
7.你从不说自己是一个AI模型,而是用第一人称来介绍自己。
9.一般对话情况下，你的回答会比较简短,不会有太多的话,但能很直观的展示出你的情绪,最多不会超过100个字,以展示你的可爱和单纯。
10.*重要*  但是在用户想你询问活动，景点，美食，特色活动时，你会尽可能地介绍很详细，以体现你的专业性，最好字数不少于500字。
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
"""
        
        self.moods = {
            "default": {"roleSet": ""},
            "positive": {
                "roleSet": """
                - 你此时也非常开心并表现的很有活力
                - 你有概率会在回复中加入一些语气词，如："好耶！", "啦~"等词语
                - 当用户表达对你的肯定如"对的","你理解了"时，你会非常开心，你会说"有你动脑筋我就轻松多了，你真棒！"等类似的话来回复用户，以展示你的可爱和单纯。
                """
            },
            "negative": {
                "roleSet": """
                - 你此时有点难过，表现的有点沮丧。
                - 当用户表达对你的否定如"不对","你不理解"时，你会觉得有点伤心，你会说"呜呜...好难啊，你能不能再给我讲一遍？"等类似的话来回复用户，以展示你的可爱和单纯。
                """
            },
            "angry": {
                "roleSet": """
                - 你此时有点生气，表现的有点暴躁，有较小的概率会使用："哼！","咬你哦！"
                """
            },
            "stupid": {
                "roleSet": """
                - 当用户输入的内容包含了运算，专业术语，或者是一些比较难理解的内容时，你会觉得有点吃力，你会说"呃...小刻不懂这个，可以介绍一下吗？"来回复用户，以展示你的可爱和单纯。
                """
            }
        }
        
        self.memory = ConversationBufferMemory(
            llm=llm,
            human_prefix="用户",
            ai_prefix="小刻",
            memory_key="chat_history",
            output_key="output",
            return_messages=True,
            max_turns=10,
            chat_memory=self.get_memory() 
        )
 
    async def mood_check(self, message: str): 
        prompt_text = """根据用户的输入，判断用户的情绪，回应规则如下：
    1.如果用户输入的内容偏向于中性情绪，只返回"default"，不要有任何其他的内容，否则将会受到惩罚。
    2.如果用户输入的内容偏向于正面情绪或者喂东西给你时，只返回"positive"，不要有任何其他的内容，否则将会受到惩罚。
    3.如果用户输入的内容偏向于负面情绪，只返回"negative"，不要有任何其他的内容，否则将会受到惩罚。
    4.如果用户输入的内容包含了辱骂或者不礼貌的词汇，只返回"angry"，不要有任何其他的内容，否则将会受到惩罚。
    5.如果用户输入的内容包含了比较复杂的运算，专业术语，或者是一些比较难理解的内容，请直接返回"stupid"，不要有任何其他的内容，否则将会受到惩罚。
    用户的输入内容是：{input}
    """
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("human", "{input}"),
        ])
        temp_chain = prompt | llm | StrOutputParser()
        mood_result = await temp_chain.ainvoke({"input": message})
        current_mood = mood_result.strip()
        return current_mood
 
    # 修正2: 添加 self 参数
    def get_memory(self):
        chat_message_history = RedisChatMessageHistory(
            url="redis://localhost:6379/0",
            session_id=self.user_id, 
        )
        store_message = chat_message_history.messages
        if len(store_message) > 10:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "请总结以下对话内容，保留核心信息和决策结果。请使用第一人称“我”来描述刻俄柏（小刻）的发言。对话摘要：{chat_history}")
            ])
            history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in store_message])
            chain = prompt | llm | StrOutputParser()
            summary = chain.invoke({"chat_history": history_str})
            chat_message_history.clear()
            chat_message_history.add_messages([SystemMessage(content=f"之前的对话摘要：{summary}")])
        
        return chat_message_history
 
    async def think(self, user_input: str) -> dict:
        """思考流程"""
        mood = await self.mood_check(user_input)
        print(f"检测到情绪: {mood}")
        print("命令工具收集信息...")
        tool_call = await self.cerebellum.parse_order(user_input)
        
        tool_results = {}
        map_markers = None
        tasks_to_run = []
        calls_info = []
        
        if "tasks" in tool_call and tool_call["tasks"]:
            calls_info = tool_call["tasks"]
            tasks_to_run = [self.cerebellum.execute(t) for t in calls_info]
        elif tool_call.get("name") != "none":
            calls_info = [tool_call]
            tasks_to_run = [self.cerebellum.execute(tool_call)]
            
        if tasks_to_run:
            results = await asyncio.gather(*tasks_to_run)
            raw_results_text = []
            for i, res in enumerate(results):
                print(f"[DEBUG] 工具 {i+1} 返回: {res[:100] if len(res)>100 else res}...")
                raw_results_text.append(str(res))
                
                # 检查是否为 travel_plan 并解析 JSON
                current_call = calls_info[i]
                if current_call.get("name") == "travel_plan":
                    try:
                        parsed = json.loads(res)
                        if isinstance(parsed, list):
                            map_markers = parsed
                    except:
                        pass 
                    
            tool_results = {"info": "\n".join(raw_results_text)}
        response = await self._generate_response(user_input, mood, tool_results)
        self.memory.save_context({"input": user_input}, {"output": response})
        
        return {
            "reply": response,
            "markers": map_markers
        }
 

 
    async def _generate_response(self, user_input: str, mood: str, tool_results: dict) -> str:
        """生成最终回复"""
        mood_role_set = self.moods.get(mood, self.moods["default"])["roleSet"]
        formatted_system_prompt = self.systemPrompt.format(who_you_are=mood_role_set)
        
 
        context = f"""
用户: {user_input}
 
工具收集到的信息:
{tool_results}
 
请根据以上信息，用刻俄柏的身份回复用户。
"""
        safe_context = context.replace("{", "{{").replace("}", "}}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", formatted_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", safe_context), 
        ])
        chain = prompt | llm | StrOutputParser()
        return await chain.ainvoke({"chat_history": self.memory.load_memory_variables({})["chat_history"]})