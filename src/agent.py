from typing import Dict, Any, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables.base import RunnableSerializable

class CustomAgentExecutor:
    chat_history: List[BaseMessage]

    def __init__(self, prompt: Any, llm: Any, tools: List[Any], name2tool: Dict[str, Any], max_iterations: int = 3):
        self.prompt = prompt
        self.llm = llm
        self.tools = tools
        self.name2tool = name2tool  # Store the tool mapping on the instance
        self.chat_history = []
        self.max_iterations = max_iterations
        self.agent: RunnableSerializable = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: x.get("agent_scratchpad", [])
            }
            | prompt
            | llm.bind_tools(tools, tool_choice="any")
        )

    def invoke(self, input: str) -> Dict[str, Any]:
        count = 0
        agent_scratchpad = []
        tool_out = None
        
        while count < self.max_iterations:
            tool_call = self.agent.invoke({
                "input": input,
                "chat_history": self.chat_history,
                "agent_scratchpad": agent_scratchpad
            })
            agent_scratchpad.append(tool_call)
            
            if not getattr(tool_call, "tool_calls", []):
                print(f"Warning: No tool calls in response at iteration {count}")
                break
                
            tool_name = tool_call.tool_calls[0]["name"]
            tool_args = tool_call.tool_calls[0]["args"]
            tool_call_id = tool_call.tool_calls[0]["id"]
            
            tool_out = self.name2tool[tool_name](**tool_args)
            
            if not isinstance(tool_out, dict) or "answer" not in tool_out:
                print(f"Warning: Tool {tool_name} returned invalid output: {tool_out}")
                tool_out = {"answer": str(tool_out)}
            
            tool_exec = ToolMessage(
                content=str(tool_out),
                tool_call_id=tool_call_id
            )
            agent_scratchpad.append(tool_exec)
            
            print(f"{count}: {tool_name}({tool_args})")
            count += 1
            
            if tool_name == "final_answer":
                break
        
        if not tool_out or not isinstance(tool_out, dict) or "answer" not in tool_out:
            tool_out = {
                "answer": "I apologize, but I wasn't able to complete the request successfully.",
                "error": "No valid tool output generated"
            }
        
        final_answer = tool_out["answer"]
        self.chat_history.extend([
            HumanMessage(content=input),
            AIMessage(content=final_answer)
        ])
        
        return tool_out