import asyncio
import json
import os
import sys

# ==========================================
# 1. 设置环境变量 (请替换为你自己的 Key)
# ==========================================
# 为了测试方便，直接在这里设置环境变量
# 实际生产中请配置在系统环境变量或 .env 文件中

# ==========================================
# 2. 导入 tools 模块
# ==========================================
try:
    # 假设你的代码保存在 tools.py 中
    from tools import travel_plan
    print("✅ 成功导入 tools 模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保当前目录下存在 'tools.py' 文件")
    sys.exit(1)

# ==========================================
# 3. 定义测试主函数
# ==========================================
async def main():
    print("\n" + "="*30)
    print("🚀 开始测试 travel_plan 工具")
    print("="*30)

    # 准备测试数据
    # 注意：travel_plan 被 @tool 装饰器包装过，且参数名为 query
    # 所以传入的字典必须包含 "query" 这个 key
    test_input = {
        "query": {
            "query_activities": ["故宫博物院", "南锣鼓巷"],  # 测试活动列表
            "query_weather": "北京今天天气晴朗，气温20-28度，微风" # 测试天气信息
        }
    }

    print(f"📝 发送测试数据: \n{json.dumps(test_input, ensure_ascii=False, indent=2)}\n")
    print("⏳ 正在调用智谱AI搜索及规划模型，请稍候...\n")

    try:
        # ==========================================
        # 4. 调用工具
        # ==========================================
        # 因为 travel_plan 是异步函数且被 @tool 装饰，使用 ainvoke 调用
        result = await travel_plan.ainvoke(test_input)

        # ==========================================
        # 5. 解析并打印结果
        # ==========================================
        print("-" * 30)
        print("✅ 测试完成！返回结果如下：")
        print("-" * 30)

        # 尝试格式化输出 JSON
        try:
            parsed_result = json.loads(result)
            print(json.dumps(parsed_result, ensure_ascii=False, indent=2))
            
            # 简单的断言检查
            assert isinstance(parsed_result, list), "结果应该是一个列表"
            if parsed_result:
                 assert "location" in parsed_result[0], "结果应包含 location 字段"
                 print("\n🎉 结果格式验证通过！")
            else:
                 print("\n⚠️ 返回结果为空列表，可能是搜索无结果或模型未生成内容。")
                 
        except json.JSONDecodeError:
            print("⚠️ 返回结果不是有效的 JSON 格式:")
            print(result)

    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

# ==========================================
# 6. 运行测试
# ==========================================
if __name__ == "__main__":
    # 检查 Key 是否已替换
    if os.getenv("ZHIPU_API_KEY") == "YOUR_ZHIPU_API_KEY_HERE":
        print("⚠️ 警告: 请先在代码中设置你的真实 ZHIPU_API_KEY")
    else:
        asyncio.run(main())