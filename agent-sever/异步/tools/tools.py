import asyncio
import json
import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from zhipuai import ZhipuAI
from datetime import datetime
#智谱AI搜索
zhipuai_client= ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))


async def web_search(query):
    def sync_search():
        try:
            resp = zhipuai_client.web_search.web_search(
                search_engine='search_pro',
                search_query=query,
            )
            if hasattr(resp, 'search_result') and resp.search_result:
                return "\n\n".join([d.content for d in resp.search_result])
            else:
                return "没有搜索到任何结果"
        except Exception as e:
            print(e)
            return f"工具调用出错: {str(e)}"
    
    return await asyncio.to_thread(sync_search)


@tool()
def get_now_time() -> str:
    """
    this is a tool which can only be used by the agent to get the current time and there is on need for args

    Returns:
        return the result in the format of str.
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


@tool()
async def get_weather(query: str) -> str:
    """
    this is a tool which can only be used by the agent to search the wether including temperature,visbility,UV index and wind speed

    Args:
        query (str): the city name  
    Returns:
        return the result in the format of str.
    """
    return await web_search(f"{query}的天气")


@tool()
async def get_activity(query: str) -> str:
    """
    this is a tool which can only be used by the agent to search for local events and activities including famous attractions, museums, parks, and cultural events.

    Args:
        query (str): the city name and key word 
    Returns:
        return the result in the format of str.
    """
    return await web_search(query)



@tool()
async def travel_plan(query:dict):
    """
    Intially, you can use this tool to the more detailed information about activities online.
    Secondly,you will according to the information formulate the optimal strategy.
    
    Args:
        query_activities(list): the activity ,query_weather(str):the city's weather
    Return:
        return the result in the format of str
    """
    query_activities = query.get("query_activities", [])
    query_weather = query.get("query_weather", "")
    
    try:
        tasks = [web_search(act) for act in query_activities]
        resp = await asyncio.gather(*tasks)

    except Exception as e:
        print(e)
        return  f"工具调用出错: {str(e)}"
    
    detial_activities=''.join(resp)
    zhi_agent = ChatOpenAI(model="glm-4-flash",
                           api_key=os.getenv("ZHIPU_API_KEY"),
                           temperature=0,
                           base_url="https://open.bigmodel.cn/api/paas/v4/"
                           )
    prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的旅行规划师。*重要*请根据给定的活动和天气信息，为每一个活动生成一个详细的行程建议。
请严格按照以下JSON数组格式输出，不要包含Markdown标记：
[
  {{
    "location": "具体地点名称（必须是真实存在的地名，用于地图搜索）",
    "title": "活动标题或简短主题",
    "desc": "详细的时间安排、注意事项和游玩建议"
  }},
  {{
    "location": "另一个地点名称",
    "title": "活动标题",
    "desc": "详细描述"
  }}
]
 
注意：
1. 必须为数组格式。
2. 每个对象必须包含 location, title, desc 三个字段。
3. location 必须具体到可以在地图上搜索到的名称（如"天坛公园"而不是"公园"）。"""),
    ("human", "活动: {{detial_activities}}\n天气: {{query_weather}}")
    ])
    chain = prompt | zhi_agent
#单独一个模型进行总结
    try:
            response = await chain.ainvoke({
                "query_activities": ", ".join(query_activities),
                "detial_activities": detial_activities, 
                "query_weather": query_weather
            })
            
            content = response.content.strip()
            # 清洗 Markdown 标记
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            # 验证 JSON 有效性
            parsed_data = json.loads(content)
            if isinstance(parsed_data, list):
                return json.dumps(parsed_data, ensure_ascii=False)
            else:
                # 如果模型返回了单个对象，包装成列表
                return json.dumps([parsed_data], ensure_ascii=False)
                
    except json.JSONDecodeError:
            error_json = [{"location": "解析错误", "title": "生成失败", "desc": "模型未能生成有效的JSON格式"}]
            return json.dumps(error_json, ensure_ascii=False)
    except Exception as e:
        print(f"❌ 发生未捕获的异常: {e}") # 打印错误到控制台
        import traceback
        traceback.print_exc() # 打印完整的堆栈信息
        # 返回包含错误信息的JSON，方便调试
        error_info = [{"location": "系统错误", "title": "工具执行异常", "desc": str(e)}]
        return json.dumps(error_info, ensure_ascii=False)

