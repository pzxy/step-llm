# 第四章 智能体经典范式构建



ReAct (Reasoning and Acting)： 边思考边做
Plan-and-Solve： 三思而后行
Reflection：反思

## 4.1 环境准备与基础工具定义
```py
pip install openai python-dotenv
```
调用函数 [llm_client.py](./llm_client.py)

## 4.2 ReAct
ReAct (Reason + Act),“思考-行动-观察” 循环

### 4.2.1 ReAct 的工作流程
 Thought(思考) -> Action（行动） -> Observation（观察结果） 的循环

![react](./react.png)

### 4.2.2 工具（Tools）的定义与实现

```bash
pip install google-search-results

```

去 [serpapi](https://serpapi.com/) 注册账号，获取key，填到 SERPAPI_API_KEY

(1)实现搜索工具的核心逻辑
- 名称 (Name)
- 描述 (Description)：这是整个机制中最关键的部分，因为大语言模型会依赖这段描述来判断何时使用哪个工具
- 执行逻辑 (Execution Logic)：执行任务的函数

search函数[tools.py](./tools.py)

(2)构建通用的工具执行器
统一的管理器来注册和调度这些工具

[ToolExecutor](./tools.py)类


## 4.2.3 ReAct 智能体的编码实现
1. 系统提示词设计
上面定义了智能体和LLM交互规范
- 角色定义
- 工具清单 ({tools})
- 格式规约 (Thought/Action)
- 动态上下文 ({question}/{history})
2. 核心循环的实现

[ReAct.py](./demo0423_ ReAct.py) 的核心是一个循环，它不断地“格式化提示词 -> 调用LLM -> 执行动作 -> 整合结果”，直到任务完成或达到最大步数限制。

