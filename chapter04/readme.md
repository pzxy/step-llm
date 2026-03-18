# 第四章 智能体经典范式构建



ReAct (Reasoning and Acting)： 边思考边做
Plan-and-Solve： 三思而后行
Reflection：反思

## 4.1 环境准备与基础工具定义
```py
pip install openai python-dotenv
```
调用函数 [demo0401.py](./demo0401.py)

## 4.2 ReAct
ReAct (Reason + Act),“思考-行动-观察” 循环

### 4.2.1 ReAct 的工作流程
 Thought(思考) -> Action（行动） -> Observation（观察结果） 的循环


### 4.2.2 工具（Tools）的定义与实现

```bash
pip install google-search-results

```

去 [serpapi](https://serpapi.com/) 注册账号，获取key，填到 SERPAPI_API_KEY

(1)实现搜索工具的核心逻辑
- 名称 (Name)
- 描述 (Description)：这是整个机制中最关键的部分，因为大语言模型会依赖这段描述来判断何时使用哪个工具
- 执行逻辑 (Execution Logic)：执行任务的函数

search函数[demo0422.py](./demo0422.py)

(2)构建通用的工具执行器
统一的管理器来注册和调度这些工具

[ToolExecutor](./demo0422.py)类



